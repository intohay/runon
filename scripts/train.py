import os
import sys

# srcディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import print_gpu_info

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import PersonaDataset
from src.model.model import PersonaModel
from src.training.trainer import PersonaTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # GPUの確認
    print_gpu_info()

    # モデルとトークナイザーの準備
    model_handler = PersonaModel(
        model_name=cfg.model.model_name,
        max_seq_length=cfg.data.max_length,
        use_4bit=cfg.model.use_4bit,
    )

    # モデルをロード
    model, tokenizer = model_handler.setup_model()

    # LoRAの適用
    model = model_handler.apply_lora(model, cfg.model.lora)

    # データセットの準備
    dataset_handler = PersonaDataset(
        train_file=cfg.data.train_file,
        eval_file=cfg.data.eval_file,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        debug_mode=cfg.debug.enabled,
    )

    # データセットを前処理
    dataset = dataset_handler.prepare_dataset()

    print("trainデータのサンプル:", dataset["train"][:3])
    print("validationデータのサンプル:", dataset["validation"][:3])

    # トレーナーの設定と学習
    trainer = PersonaTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        training_config=cfg.training,
        output_dir=cfg.training.output_dir,
        debug_mode=cfg.debug.enabled,
    )

    # wandbの初期化
    trainer.setup_wandb(OmegaConf.to_container(cfg, resolve=True))

    # 学習の実行
    trainer.train()

    print("学習が完了しました。")


if __name__ == "__main__":
    main()
