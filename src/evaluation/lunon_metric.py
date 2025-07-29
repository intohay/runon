import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from unsloth import FastLanguageModel

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import PersonaModel

from .base import Metric


class LUNONMetric(Metric):
    def __init__(
        self,
        base_model_name="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5",
        ft_model_name=None,
        max_seq_length=512,
        use_4bit=True,
        device=None,
    ):
        self.base_model_name = base_model_name
        self.ft_model_name = ft_model_name
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = None
        self.base_tokenizer = None
        self.ft_model = None
        self.ft_tokenizer = None

    @property
    def name(self) -> str:
        return "lunon"

    def prepare(self, reference_data: List[Dict[str, Any]]):
        """モデルを準備する"""
        if self.ft_model_name is None:
            raise ValueError("ft_model_name must be specified for LUNON metric")

        base = PersonaModel(self.base_model_name, self.max_seq_length, self.use_4bit)
        ft = PersonaModel(self.ft_model_name, self.max_seq_length, self.use_4bit)
        self.base_model, self.base_tokenizer = base.setup_model()
        self.ft_model, self.ft_tokenizer = ft.setup_model()
        FastLanguageModel.for_inference(self.base_model)
        FastLanguageModel.for_inference(self.ft_model)

    def compute(self, hypothesis, reference_data: List[Dict[str, Any]] = None) -> float:
        """単一のテキストに対してLUNONスコアを計算"""
        if self.base_model is None or self.ft_model is None:
            raise RuntimeError("Models not prepared. Call prepare() first.")

        # hypothesisがdict形式（messagesを含む）の場合の処理
        if isinstance(hypothesis, dict) and "messages" in hypothesis:
            # prefix（user）とcontinuation（assistant）を抽出
            prefix = ""
            continuation = ""
            for msg in hypothesis["messages"]:
                if msg.get("role") == "user":
                    prefix = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    continuation = msg.get("content", "")

            if prefix and continuation:
                # 条件付きSLORスコアを計算
                return calc_conditional_slor_score_batch(
                    [prefix],
                    [continuation],
                    self.base_model,
                    self.base_tokenizer,
                    self.ft_model,
                    self.ft_tokenizer,
                    self.device,
                )[0]
            else:
                # prefixまたはcontinuationが見つからない場合は全体を使用
                combined_text = ""
                for msg in hypothesis["messages"]:
                    if msg.get("role") in ["user", "assistant"]:
                        content = msg.get("content", "")
                        if content:
                            combined_text += content
                return calc_slor_score_batch(
                    [combined_text],
                    self.base_model,
                    self.base_tokenizer,
                    self.ft_model,
                    self.ft_tokenizer,
                    self.device,
                )[0]
        else:
            # 文字列の場合は従来の方法
            text = str(hypothesis) if hypothesis else ""
            return calc_slor_score_batch(
                [text],
                self.base_model,
                self.base_tokenizer,
                self.ft_model,
                self.ft_tokenizer,
                self.device,
            )[0]

    def aggregate(self, scores: List[float]) -> Dict[str, float]:
        """スコアリストを集計統計に変換"""
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        scores_array = np.array(scores)
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
        }


def calc_conditional_slor_score_batch(
    prefixes: List[str],
    continuations: List[str],
    base_model,
    base_tokenizer,
    ft_model,
    ft_tokenizer,
    device=None,
) -> List[float]:
    """prefixを条件付けしたときのcontinuationのSLORスコアを計算"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    slor_scores = []

    for prefix, continuation in zip(prefixes, continuations):
        # prefixとcontinuationを結合
        full_text = prefix + continuation

        # ベースモデル用のトークナイズ
        prefix_tokens_base = base_tokenizer(
            prefix, return_tensors="pt", add_special_tokens=False
        )
        full_tokens_base = base_tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        )
        prefix_length_base = prefix_tokens_base["input_ids"].shape[1]
        full_input_ids_base = full_tokens_base["input_ids"].to(device)

        # ファインチューニングモデル用のトークナイズ
        prefix_tokens_ft = ft_tokenizer(
            prefix, return_tensors="pt", add_special_tokens=False
        )
        full_tokens_ft = ft_tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        )
        prefix_length_ft = prefix_tokens_ft["input_ids"].shape[1]
        full_input_ids_ft = full_tokens_ft["input_ids"].to(device)

        with torch.no_grad():
            # ベースモデルの対数尤度計算
            outputs_base = base_model(input_ids=full_input_ids_base)
            logits_base = outputs_base.logits[0, :-1, :]
            target_ids_base = full_input_ids_base[0, 1:]
            continuation_logits_base = logits_base[prefix_length_base - 1 :]
            continuation_target_base = target_ids_base[prefix_length_base - 1 :]
            log_probs_base = torch.nn.functional.log_softmax(
                continuation_logits_base, dim=-1
            )
            log_prob_base = log_probs_base.gather(
                1, continuation_target_base.unsqueeze(-1)
            ).squeeze(-1)
            avg_log_prob_base = log_prob_base.mean().item()

            # ファインチューニングモデルの対数尤度計算
            outputs_ft = ft_model(input_ids=full_input_ids_ft)
            logits_ft = outputs_ft.logits[0, :-1, :]
            target_ids_ft = full_input_ids_ft[0, 1:]
            continuation_logits_ft = logits_ft[prefix_length_ft - 1 :]
            continuation_target_ft = target_ids_ft[prefix_length_ft - 1 :]
            log_probs_ft = torch.nn.functional.log_softmax(
                continuation_logits_ft, dim=-1
            )
            log_prob_ft = log_probs_ft.gather(
                1, continuation_target_ft.unsqueeze(-1)
            ).squeeze(-1)
            avg_log_prob_ft = log_prob_ft.mean().item()

            # SLORスコア（差分）
            slor_score = avg_log_prob_ft - avg_log_prob_base
            slor_scores.append(slor_score)

    return slor_scores


class LUNON:
    def __init__(
        self,
        base_model_name,
        ft_model_name,
        max_seq_length=512,
        use_4bit=True,
        device=None,
    ):
        self.base_model_name = base_model_name
        self.ft_model_name = ft_model_name
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit
        self.device = device
        self.base_model, self.base_tokenizer, self.ft_model, self.ft_tokenizer = (
            self.setup_models()
        )
        FastLanguageModel.for_inference(self.base_model)
        FastLanguageModel.for_inference(self.ft_model)

    def setup_models(self):
        base = PersonaModel(self.base_model_name, self.max_seq_length, self.use_4bit)
        ft = PersonaModel(self.ft_model_name, self.max_seq_length, self.use_4bit)
        base_model, base_tokenizer = base.setup_model()
        ft_model, ft_tokenizer = ft.setup_model()
        return base_model, base_tokenizer, ft_model, ft_tokenizer

    @property
    def name(self) -> str:
        return "lunon"


def calc_slor_score_batch(
    texts: List[str],
    base_model,
    base_tokenizer,
    ft_model,
    ft_tokenizer,
    device=None,
) -> List[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # トークナイズ（バッチで）
    inputs_base = base_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )
    inputs_ft = ft_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids_base = inputs_base["input_ids"].to(device)
    input_ids_ft = inputs_ft["input_ids"].to(device)
    labels_base = input_ids_base.clone()
    labels_ft = input_ids_ft.clone()

    with torch.no_grad():
        # baseモデル
        outputs_base = base_model(input_ids=input_ids_base)
        logits_base = outputs_base.logits[:, :-1, :]
        target_base = labels_base[:, 1:]
        log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)
        log_prob_base = log_probs_base.gather(2, target_base.unsqueeze(-1)).squeeze(-1)
        sum_log_prob_base = log_prob_base.sum(dim=1)
        # パディングを除いた長さ
        lengths = (target_base != base_tokenizer.pad_token_id).sum(dim=1)

        # ftモデル
        outputs_ft = ft_model(input_ids=input_ids_ft)
        logits_ft = outputs_ft.logits[:, :-1, :]
        target_ft = labels_ft[:, 1:]
        log_probs_ft = torch.nn.functional.log_softmax(logits_ft, dim=-1)
        log_prob_ft = log_probs_ft.gather(2, target_ft.unsqueeze(-1)).squeeze(-1)
        sum_log_prob_ft = log_prob_ft.sum(dim=1)

    # 各サンプルごとにSLORスコア
    slor_scores = ((sum_log_prob_ft - sum_log_prob_base) / lengths).tolist()
    return slor_scores


if __name__ == "__main__":
    text = [
        """おはぴよ🐥

今日も寒いねー

マンボウちゃんが住んでるところってどんな感じ？？

東京より暖かい？

なんか楽しそうな場所だよね〜
今度行ってみたいな😚

でも、さっきの質問に答えてない！
私の住んでるのは東京都内なの～
だから今回は東京と同じくらい寒いでしょう！！"""
    ]

    base_model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-v0.2"
    ft_model_name = (
        "models/hiyori/llama-3.1-swallow-8b/2025-05-26_15-41-44/checkpoints/"
    )
    max_seq_length = 512
    use_4bit = True
    device = "cuda"
    print(
        calc_slor_score_batch(
            text, base_model_name, ft_model_name, max_seq_length, use_4bit, device
        )
    )
