import json
import os
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import EarlyStoppingCallback, TrainerCallback
from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import wandb


class PerplexityCallback(TrainerCallback):
    """eval_lossからperplexityを計算し、ログに記録するコールバック"""

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """評価時にperplexityを計算して記録"""
        if logs is not None and "eval_loss" in logs:
            perplexity = np.exp(logs["eval_loss"])
            logs["eval_perplexity"] = perplexity
            print(
                f"Eval Loss: {logs['eval_loss']:.4f}, Eval Perplexity: {perplexity:.4f}"
            )

            # wandbにも記録
            if wandb.run is not None:
                wandb.log(
                    {"eval_loss": logs["eval_loss"], "eval_perplexity": perplexity}
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """ログ時にtraining lossも表示"""
        if logs is not None and "loss" in logs:
            train_perplexity = np.exp(logs["loss"])
            print(
                f"Train Loss: {logs['loss']:.4f}, Train Perplexity: {train_perplexity:.4f}"
            )


class PersonaTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any,
        training_config: Dict[str, Any],
        output_dir: str,
        debug_mode: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.output_dir = output_dir
        self.debug_mode = debug_mode

    def setup_wandb(self, config: Dict[str, Any]) -> None:
        """Weights & Biasesをセットアップします"""
        # 設定からrun名を取得するか、デフォルト値を使用
        run_name = config["wandb"].get("run_name", None)

        # run名が指定されていない場合は、タイムスタンプ付きのデフォルト名を生成することもできる
        if not run_name and "experiment_name" in config:
            run_name = f"{config['experiment_name']}"

        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            mode=config["wandb"]["mode"],
            name=run_name,  # run名を指定
            config=config,
        )

    def compute_metrics(self, eval_pred):
        """評価メトリクスを計算します（perplexity）"""
        # eval_predからはlossを直接計算できないので、
        # eval_lossからperplexityを計算するのはコールバックで行います
        return {}

    def setup_training_args(self) -> UnslothTrainingArguments:
        """トレーニング引数をセットアップします"""
        return UnslothTrainingArguments(
            per_device_train_batch_size=self.training_config["batch_size"],
            per_device_eval_batch_size=self.training_config["batch_size"],
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
            warmup_ratio=self.training_config["warmup_ratio"],
            num_train_epochs=1
            if self.debug_mode
            else self.training_config["num_train_epochs"],
            learning_rate=self.training_config["learning_rate"],
            embedding_learning_rate=self.training_config["embedding_learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1
            if self.debug_mode
            else self.training_config["logging_steps"],
            optim="adamw_8bit",
            weight_decay=self.training_config["weight_decay"],
            lr_scheduler_type="linear",
            seed=self.training_config["seed"],
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=self.training_config["load_best_model_at_end"],
            save_total_limit=self.training_config.get(
                "save_total_limit", None
            ),  # checkpointの保存数制限
            metric_for_best_model=self.training_config.get(
                "metric_for_best_model", "eval_perplexity"
            ),  # ベストモデルの評価指標をperplexityに変更
            greater_is_better=self.training_config.get(
                "greater_is_better", False
            ),  # perplexityは低いほど良い
            report_to="wandb",
        )

    def save_best_metric(self, best_perplexity: float) -> None:
        """最高のperplexityをローカルファイルに保存"""
        metrics_file = os.path.join(self.output_dir, "training_metrics.json")

        # 既存のメトリクスファイルがあれば読み込み
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # 最高のperplexityを追加
        metrics["best_perplexity"] = best_perplexity

        # DictConfigを通常のdictに変換
        if isinstance(self.training_config, DictConfig):
            training_config_dict = OmegaConf.to_container(
                self.training_config, resolve=True
            )
        else:
            training_config_dict = self.training_config

        metrics["training_config"] = training_config_dict

        # ファイルに保存
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Best perplexity saved to: {metrics_file}")

    def train(self) -> None:
        """モデルをトレーニングします"""
        training_args = self.setup_training_args()

        # トレーナーの初期化
        trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.training_config["max_length"],
            dataset_num_proc=2,
            compute_metrics=None,  # compute_metricsを無効化
            args=training_args,
        )

        # Perplexityコールバックを追加
        trainer.add_callback(PerplexityCallback())

        # Early Stoppingコールバックを追加
        if not self.debug_mode:  # デバッグモードでは無効
            early_stopping_patience = self.training_config.get(
                "early_stopping_patience", 3
            )
            early_stopping_threshold = self.training_config.get(
                "early_stopping_threshold", 0.01
            )
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold,
                )
            )

        if self.tokenizer.chat_template is not None:
            trainer = train_on_responses_only(
                trainer,
                # ↓テンプレートに応じてユーザ開始トークン／アシスタント開始トークンを明示
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )

        # 学習開始前のGPUメモリ使用量
        if torch.cuda.is_available():
            print("\nGPU Memory before training:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # 学習
        trainer.train()

        # 学習終了後のGPUメモリ使用量
        if torch.cuda.is_available():
            print("\nGPU Memory after training:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # 最高のperplexityを計算（trainer.stateから取得）
        best_loss = None
        best_perplexity = None

        # trainer.stateから最高のeval_lossを取得
        if (
            hasattr(trainer.state, "best_metric")
            and trainer.state.best_metric is not None
        ):
            # metric_for_best_modelがeval_perplexityの場合
            if training_args.metric_for_best_model == "eval_perplexity":
                best_perplexity = trainer.state.best_metric
                best_loss = np.log(best_perplexity)
            else:
                # metric_for_best_modelがeval_lossの場合
                best_loss = trainer.state.best_metric
                best_perplexity = np.exp(best_loss)

        # log_historyからも確認
        if best_perplexity is None and hasattr(trainer.state, "log_history"):
            eval_perplexities = []
            for log in trainer.state.log_history:
                if "eval_perplexity" in log:
                    eval_perplexities.append(log["eval_perplexity"])

            if eval_perplexities:
                best_perplexity = min(eval_perplexities)
                best_loss = np.log(best_perplexity)

        if best_perplexity is not None:
            print(f"\nBest Loss: {best_loss:.4f}")
            print(f"Best Perplexity: {best_perplexity:.4f}")

            # ローカルファイルに保存
            self.save_best_metric(best_perplexity)

            # wandbにも記録
            if wandb.run is not None:
                wandb.log({"best_loss": best_loss, "best_perplexity": best_perplexity})

        # モデルとトークナイザーを保存
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
