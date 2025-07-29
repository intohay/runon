#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import os
import sys
from typing import Any, Dict, List, Type

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.base import Metric
from src.evaluation.base_nll import BaseNLLMetric
from src.evaluation.bertscore_metric import BERTScoreMetric
from src.evaluation.blue_metric import BLEUMetric
from src.evaluation.chatgpt_comparative_metric import ChatGPTComparativeMetric
from src.evaluation.chatgpt_metric import ChatGPTMetric
from src.evaluation.ipsi_metric import IPSIMetric
from src.evaluation.length_metric import LengthMetric
from src.evaluation.lunon_metric import LUNONMetric
from src.evaluation.lunon_poly_metric import LUNONPolyMetric
from src.evaluation.nll import NLLMetric
from src.evaluation.persona_slor import PersonaSLORMetric
from src.evaluation.personaclr_metric import PersonaCLRMetric

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_train_data(file_path: str) -> List[Dict[str, Any]]:
    """
    訓練データをJSONまたはJSONLファイルから読み込み、各メッセージの情報を返します。

    Args:
        file_path: 訓練データのJSONまたはJSONLファイルのパス

    Returns:
        処理されたメッセージのリスト
    """
    # ファイル拡張子を確認してデータの読み込み方法を決定
    if file_path.lower().endswith(".jsonl"):
        # JSONLファイルの場合：各行がJSONオブジェクト
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 空行をスキップ
                    data.append(json.loads(line))
    else:
        # JSONファイルの場合：従来の方法
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # アシスタントのメッセージを抽出して処理
    processed_data = []
    for item in data:
        for message in item["messages"]:
            if message["role"] == "assistant":
                processed_data.append(
                    {
                        "content": message["content"],
                        "metadata": {
                            "original_item": item
                        },  # 元のデータを保持（将来の拡張用）
                    }
                )

    return processed_data


def get_available_metrics() -> Dict[str, Type[Metric]]:
    """利用可能な評価指標のディクショナリを返す"""
    return {
        "bleu": BLEUMetric,
        "length": LengthMetric,
        "personaclr": PersonaCLRMetric,
        "lunon_similar": PersonaSLORMetric,
        "nll": NLLMetric,
        "base_nll": BaseNLLMetric,
        "ipsi": IPSIMetric,
        "lunon": LUNONMetric,
        "lunon_poly": LUNONPolyMetric,
        "chatgpt": ChatGPTMetric,
        "chatgpt_comparative": ChatGPTComparativeMetric,
        "bertscore": BERTScoreMetric,
        # "psprob": PSProbMetric,
        # "ptsal": PTSalMetric,
        # 将来的に他の指標を追加する場合はここに追加
        # "rouge": ROUGEMetric,
    }


def save_individual_scores(
    individual_scores: List[Dict[str, Any]],
    output_dir: str,
    file_name: str,
    format: str = "json",
):
    """個々のスコアを指定された形式で保存する（既存結果とマージ）"""
    # evalフォルダを作成
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if format == "json":
        output_path = os.path.join(eval_dir, f"{file_name}_individual_scores.json")

        # 既存ファイルを読み込み
        existing_scores = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_scores = json.load(f)
            except Exception as e:
                print(f"既存の個別スコア読み込みに失敗: {e}")

        # 既存スコアと新スコアをマージ
        merged_scores = merge_individual_scores(existing_scores, individual_scores)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_scores, f, ensure_ascii=False, indent=2)
        print(f"個々のスコアを保存しました: {output_path}")

    elif format == "csv":
        output_path = os.path.join(eval_dir, f"{file_name}_individual_scores.csv")

        # 既存ファイルを読み込み
        existing_df = None
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
            except Exception as e:
                print(f"既存の個別スコア読み込みに失敗: {e}")

                # 新しいスコアを辞書形式に変換
        scores_dict = {}
        for score in individual_scores:
            index = score["index"]
            if index not in scores_dict:
                scores_dict[index] = {
                    "index": index,
                    "prefix": score.get("prefix", ""),
                    "continuation": score.get("continuation", ""),
                }

            # スコアの処理
            metric_name = score["metric"]
            score_value = score["score"]

            # スコアが辞書形式の場合はスコア値のみを抽出
            if isinstance(score_value, dict) and "score" in score_value:
                actual_score = score_value["score"]
            else:
                actual_score = score_value

            scores_dict[index][metric_name] = actual_score

        new_df = pd.DataFrame(list(scores_dict.values()))

        # 既存データとマージ（新しい値で上書き）
        if existing_df is not None:
            # indexをキーにしてマージ（新しい値を優先）
            merged_df = (
                new_df.set_index("index")
                .combine_first(existing_df.set_index("index"))
                .reset_index()
            )
            # 新しい指標のカラムを追加、既存指標の上書きを報告
            for col in new_df.columns:
                if col not in existing_df.columns:
                    print(f"  新しい指標 {col} を追加しました")
                elif col not in ["index", "prefix", "continuation"]:
                    print(f"  既存の指標 {col} を上書きしました")
        else:
            merged_df = new_df

        # カラムの順序を調整（index, prefix, continuation, 手法名の順序）
        fixed_order = ["index", "prefix", "continuation"]

        if existing_df is not None:
            # 既存のカラム順序を維持し、新しいカラムのみ右端に追加
            existing_cols = [
                col for col in existing_df.columns if col in merged_df.columns
            ]
            new_cols = [
                col
                for col in merged_df.columns
                if col not in existing_df.columns and col not in fixed_order
            ]
            final_order = existing_cols + new_cols
        else:
            # 新規作成の場合は、index, hypothesisを先頭に、残りはアルファベット順
            other_cols = [col for col in merged_df.columns if col not in fixed_order]
            final_order = fixed_order + sorted(other_cols)

        merged_df = merged_df[final_order]

        merged_df.to_csv(output_path, index=False)
        print(f"個々のスコアを保存しました: {output_path}")
    else:
        print(f"未サポートの出力形式: {format}")


def merge_individual_scores(
    existing_scores: List[Dict], new_scores: List[Dict]
) -> List[Dict]:
    """個別スコアをマージする"""
    # indexをキーにした辞書を作成
    merged_dict = {}

    # 既存スコアを辞書に追加
    for score in existing_scores:
        index = score["index"]
        if index not in merged_dict:
            merged_dict[index] = {
                "index": index,
                "prefix": score.get("prefix", ""),
                "continuation": score.get("continuation", ""),
            }

        # スコアが辞書形式の場合はスコア値のみを抽出
        if isinstance(score["score"], dict) and "score" in score["score"]:
            actual_score = score["score"]["score"]
        else:
            actual_score = score["score"]

        merged_dict[index][score["metric"]] = actual_score

    # 新スコアを辞書に追加（上書き/追加）
    for score in new_scores:
        index = score["index"]
        if index not in merged_dict:
            merged_dict[index] = {
                "index": index,
                "prefix": score.get("prefix", ""),
                "continuation": score.get("continuation", ""),
            }

        # スコアが辞書形式の場合はスコア値のみを抽出
        if isinstance(score["score"], dict) and "score" in score["score"]:
            actual_score = score["score"]["score"]
        else:
            actual_score = score["score"]

        merged_dict[index][score["metric"]] = actual_score

    # 辞書をリスト形式に変換
    result = []
    for index, score_data in merged_dict.items():
        prefix = score_data.pop("prefix", "")
        continuation = score_data.pop("continuation", "")
        score_data.pop("index")
        for metric, score in score_data.items():
            result.append(
                {
                    "index": index,
                    "prefix": prefix,
                    "continuation": continuation,
                    "metric": metric,
                    "score": score,
                }
            )

    return sorted(result, key=lambda x: (x["index"], x["metric"]))


def evaluate_outputs(
    train_data_path: str,
    eval_dir: str = None,
    metrics: List[Metric] = None,
    output_format: str = "json",
    eval_file: str = None,
    labels: Dict[str, str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    eval_dir内のCSVファイル、または単一のCSVファイルの生成結果を複数の指標で評価します。

    Args:
        train_data_path: 訓練データのJSONファイルパス
        eval_dir: 評価対象CSVファイルが格納されているディレクトリパス
        metrics: 使用する評価指標のリスト
        output_format: 出力形式 ("json" または "csv")
        eval_file: 評価対象の単一CSVファイルパス

    Returns:
        ファイル名ごと、指標ごとの評価結果を格納した階層的辞書
    """
    # 訓練データの読み込み
    if train_data_path is not None:
        print(f"訓練データを読み込んでいます: {train_data_path}")
        reference_data = load_train_data(train_data_path)
        print(f"参照データ数: {len(reference_data)}")
    else:
        reference_data = None

    # 各指標の初期化
    for metric in metrics:
        print(f"{metric.name}指標を準備中...")
        metric.prepare(reference_data)

    # 評価対象ファイルの検索
    if eval_file:
        eval_files = [eval_file]
    elif eval_dir:
        # CSVとJSONLファイルを両方探す
        csv_files = glob.glob(os.path.join(eval_dir, "*.csv"))
        jsonl_files = glob.glob(os.path.join(eval_dir, "*.jsonl"))
        eval_files = csv_files + jsonl_files
    else:
        print("エラー: eval-dirまたはeval-fileのいずれかを指定してください")
        return {}

    if not eval_files:
        print(
            "警告: 指定されたファイル/ディレクトリにCSVまたはJSONLファイルが見つかりませんでした"
        )
        return {}

    results = {}

    # 各ファイルを評価
    for eval_file_path in eval_files:
        file_name = os.path.basename(eval_file_path)
        print(f"評価中: {file_name}")

        try:
            file_extension = os.path.splitext(eval_file_path)[1].lower()

            if file_extension == ".csv":
                # CSVファイルの処理
                df = pd.read_csv(eval_file_path)

                # 生成されたテキストの列を特定
                text_column = None
                for column in df.columns:
                    if (
                        "text" in column.lower()
                        or "content" in column.lower()
                        or "response" in column.lower()
                        or "output" in column.lower()
                    ):
                        text_column = column
                        break

                if text_column is None and len(df.columns) > 0:
                    # 列が見つからない場合は最後の列を使用
                    text_column = df.columns[-1]

                if text_column is None:
                    print(
                        f"警告: {file_name} から生成テキストの列を特定できませんでした"
                    )
                    continue

                # データを準備
                evaluation_data = []
                for idx, row in df.iterrows():
                    hypothesis = str(row[text_column])
                    if hypothesis and not pd.isna(hypothesis):
                        evaluation_data.append({"index": idx, "hypothesis": hypothesis})

            elif file_extension == ".jsonl":
                # JSONLファイルの処理
                evaluation_data = []
                with open(eval_file_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            data = json.loads(line)
                            evaluation_data.append(
                                {
                                    "index": idx,
                                    "hypothesis": data,  # JSONLの場合は辞書形式で渡す
                                }
                            )
            else:
                print(f"警告: サポートされていないファイル形式: {file_extension}")
                continue

            if not evaluation_data:
                print(f"警告: {file_name} に有効なデータが見つかりませんでした")
                continue

            file_results = {}
            individual_scores = []

            # 各指標ごとにスコアを計算
            for metric in metrics:
                metric_scores = []

                # 相対評価指標の場合は全体データを一度に処理
                if metric.name == "chatgpt_comparative":
                    # 全てのhypothesisを1つのリストとして渡す
                    all_hypotheses = [item["hypothesis"] for item in evaluation_data]
                    score = metric.compute(all_hypotheses, reference_data)

                    # 相対評価の結果から個別スコアを取得
                    for i, item in enumerate(evaluation_data):
                        hypothesis = item["hypothesis"]
                        idx = item["index"]

                        # 各example_countの結果に対して個別スコアを取得
                        individual_score = {}
                        for key, result in score.items():
                            if key.startswith("chatgpt_comparative-") and isinstance(
                                result, dict
                            ):
                                if "individual_scores" in result and i < len(
                                    result["individual_scores"]
                                ):
                                    individual_score[key] = result["individual_scores"][
                                        i
                                    ]
                                else:
                                    individual_score[key] = result.get("score", 0)
                            else:
                                individual_score[key] = result

                        metric_scores.append(individual_score)

                        # individual_scoresに各アイテムの結果を追加
                        hypothesis_str = ""
                        if isinstance(hypothesis, dict):
                            hypothesis_str = json.dumps(hypothesis, ensure_ascii=False)
                        else:
                            hypothesis_str = str(hypothesis)

                        # hypothesisからprefix/continuationを抽出
                        prefix = ""
                        continuation = ""
                        if isinstance(hypothesis, dict) and "messages" in hypothesis:
                            for msg in hypothesis["messages"]:
                                if msg.get("role") == "user":
                                    prefix = msg.get("content", "")
                                elif msg.get("role") == "assistant":
                                    continuation = msg.get("content", "")

                        # 各example_countの結果に対してindividual_scoresエントリを作成
                        for key, result in individual_score.items():
                            if key.startswith("chatgpt_comparative-"):
                                individual_scores.append(
                                    {
                                        "index": idx,
                                        "hypothesis": hypothesis_str,
                                        "prefix": prefix,
                                        "continuation": continuation,
                                        "metric": key,  # e.g., chatgpt_comparative-50
                                        "score": result,
                                    }
                                )
                else:
                    # 従来の個別処理
                    for item in tqdm(evaluation_data, desc=f"{metric.name}スコア計算"):
                        hypothesis = item["hypothesis"]
                        idx = item["index"]

                        score = metric.compute(hypothesis, reference_data)
                        metric_scores.append(score)

                        # 各仮説テキストに対する評価結果を保存
                        hypothesis_str = ""
                        if isinstance(hypothesis, dict):
                            hypothesis_str = json.dumps(hypothesis, ensure_ascii=False)
                        else:
                            hypothesis_str = str(hypothesis)

                        # ChatGPT, ChatGPT相対評価, BERTScore, BLEUの場合は複数の結果が返される
                        if isinstance(score, dict) and (
                            (
                                metric.name == "chatgpt"
                                and any(
                                    key.startswith("chatgpt-") for key in score.keys()
                                )
                            )
                            or (
                                metric.name == "bertscore"
                                and any(
                                    key.startswith("bertscore-") for key in score.keys()
                                )
                            )
                            or (
                                metric.name == "bleu"
                                and any(key.startswith("bleu-") for key in score.keys())
                            )
                        ):
                            # 複数の評価結果を処理
                            for sub_key, sub_result in score.items():
                                if sub_key.startswith(f"{metric.name}-"):
                                    prefix = ""
                                    continuation = ""

                                    # hypothesisからprefix/continuationを抽出
                                    if (
                                        isinstance(hypothesis, dict)
                                        and "messages" in hypothesis
                                    ):
                                        for msg in hypothesis["messages"]:
                                            if msg.get("role") == "user":
                                                prefix = msg.get("content", "")
                                            elif msg.get("role") == "assistant":
                                                continuation = msg.get("content", "")

                                    # スコア辞書からも試す（後方互換性のため）
                                    if isinstance(sub_result, dict):
                                        if not prefix:
                                            prefix = sub_result.get("prefix", "")
                                        if not continuation:
                                            continuation = sub_result.get(
                                                "continuation", ""
                                            )

                                    # ラベルがある場合は、サブキーの前半部分をラベルに置き換える
                                    metric_label = sub_key
                                    if labels and metric.name in labels:
                                        # e.g., chatgpt-20 → lunon-a-20
                                        parts = sub_key.split("-", 1)
                                        if len(parts) == 2:
                                            metric_label = (
                                                f"{labels[metric.name]}-{parts[1]}"
                                            )
                                        else:
                                            metric_label = labels[metric.name]

                                    individual_scores.append(
                                        {
                                            "index": idx,
                                            "hypothesis": hypothesis_str,
                                            "prefix": prefix,
                                            "continuation": continuation,
                                            "metric": metric_label,  # e.g., lunon-a-20, bertscore-50, bleu-100
                                            "score": sub_result,
                                        }
                                    )
                        else:
                            # 従来の単一結果の処理
                            prefix = ""
                            continuation = ""

                            # hypothesisからprefix/continuationを抽出
                            if (
                                isinstance(hypothesis, dict)
                                and "messages" in hypothesis
                            ):
                                for msg in hypothesis["messages"]:
                                    if msg.get("role") == "user":
                                        prefix = msg.get("content", "")
                                    elif msg.get("role") == "assistant":
                                        continuation = msg.get("content", "")

                            # スコア辞書からも試す（後方互換性のため）
                            if isinstance(score, dict):
                                if not prefix:
                                    prefix = score.get("prefix", "")
                                if not continuation:
                                    continuation = score.get("continuation", "")

                            # ラベルがある場合はラベルを使用、なければメトリクス名を使用
                            metric_label = (
                                labels.get(metric.name, metric.name)
                                if labels
                                else metric.name
                            )
                            individual_scores.append(
                                {
                                    "index": idx,
                                    "hypothesis": hypothesis_str,
                                    "prefix": prefix,
                                    "continuation": continuation,
                                    "metric": metric_label,
                                    "score": score,
                                }
                            )

                # 結果を集計
                # ラベルがある場合はラベルを使用、なければメトリクス名を使用
                metric_key = (
                    labels.get(metric.name, metric.name) if labels else metric.name
                )
                if metric_scores:
                    file_results[metric_key] = metric.aggregate(metric_scores)
                else:
                    file_results[metric_key] = {
                        "error": "有効なテキストが見つかりませんでした"
                    }

            # 各仮説テキストに対する評価結果をファイル結果に追加
            file_results["individual_scores"] = individual_scores

            # 個々のスコアを別ファイルに保存
            save_individual_scores(
                individual_scores,
                eval_dir if eval_dir else os.path.dirname(eval_file_path),
                file_name.replace(".jsonl", "").replace(".csv", ""),  # 拡張子を除去
                output_format,
            )

            results[file_name] = file_results

        except Exception as e:
            print(f"エラー: {file_name} の処理中に問題が発生しました - {str(e)}")
            results[file_name] = {"error": str(e)}

    return results


def load_existing_results(
    output_dir: str, format: str = "json"
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """既存の評価結果を読み込む"""
    existing_results = {}

    if format == "json":
        output_path = os.path.join(output_dir, "evaluation_results.json")
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                print(f"既存の評価結果を読み込みました: {output_path}")
            except Exception as e:
                print(f"既存結果の読み込みに失敗しました: {e}")

    elif format == "csv":
        output_path = os.path.join(output_dir, "eval", "evaluation_results.csv")
        if os.path.exists(output_path):
            try:
                df = pd.read_csv(output_path)
                # CSV形式から階層的辞書に再構築
                for _, row in df.iterrows():
                    file_name = row["file"]
                    metric_name = row["metric"]
                    stat_name = row["statistic"]
                    value = row["value"]

                    if file_name not in existing_results:
                        existing_results[file_name] = {}
                    if metric_name not in existing_results[file_name]:
                        existing_results[file_name][metric_name] = {}
                    existing_results[file_name][metric_name][stat_name] = value

                print(f"既存の評価結果を読み込みました: {output_path}")
            except Exception as e:
                print(f"既存結果の読み込みに失敗しました: {e}")

    return existing_results


def merge_results(existing_results: Dict, new_results: Dict) -> Dict:
    """既存結果と新しい結果をマージする（新しい指標は追加、既存指標は上書き）"""
    merged_results = existing_results.copy()

    for file_name, file_metrics in new_results.items():
        if file_name not in merged_results:
            merged_results[file_name] = {}

        for metric_name, metric_values in file_metrics.items():
            # individual_scoresは特別扱い（常に上書き）
            if metric_name == "individual_scores":
                merged_results[file_name][metric_name] = metric_values
            else:
                # 既存の指標は上書き、新しい指標は追加
                merged_results[file_name][metric_name] = metric_values
                if metric_name in existing_results.get(file_name, {}):
                    print(f"  {file_name}: {metric_name} を上書きしました")
                else:
                    print(f"  {file_name}: {metric_name} を追加しました")

    return merged_results


def save_results(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    format: str = "json",
):
    """結果を指定された形式で保存する（既存結果とマージ）"""
    os.makedirs(output_dir, exist_ok=True)

    # 既存結果を読み込み
    existing_results = load_existing_results(output_dir, format)

    # 新しい結果と既存結果をマージ
    if existing_results:
        final_results = merge_results(existing_results, results)
        print("評価結果をマージしました:")
    else:
        final_results = results
        print("新しい評価結果を作成します:")

    # numpyの数値型をPythonの組み込み型に変換
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    if format == "json":
        # JSON形式で保存
        output_path = os.path.join(output_dir, "evaluation_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                final_results,
                f,
                ensure_ascii=False,
                indent=2,
                default=convert_numpy_types,
            )
        print(f"評価結果を保存しました: {output_path}")

    elif format == "csv":
        # CSV形式で保存（階層的なデータを平坦化）
        output_path = os.path.join(output_dir, "eval", "evaluation_results.csv")
        rows = []

        for file_name, metrics in final_results.items():
            for metric_name, values in metrics.items():
                if metric_name == "individual_scores":
                    continue

                if isinstance(values, dict) and "error" not in values:
                    # ChatGPTMetricの場合、複数の結果が含まれる
                    if (
                        metric_name == "chatgpt"
                        or metric_name.startswith("chatgpt")
                        or "chatgpt" in metric_name
                    ):
                        for sub_metric_name, sub_values in values.items():
                            if (
                                isinstance(sub_values, dict)
                                and "error" not in sub_values
                            ):
                                for stat_name, value in sub_values.items():
                                    rows.append(
                                        {
                                            "file": file_name,
                                            "metric": sub_metric_name,  # chatgpt-20, chatgpt-100, chatgpt-all, label-20等
                                            "statistic": stat_name,
                                            "value": convert_numpy_types(value),
                                        }
                                    )
                    elif (
                        metric_name.startswith("chatgpt_comparative")
                        or "chatgpt_comparative" in metric_name
                    ):
                        # ChatGPT相対評価指標は複数の結果が含まれる（chatgpt_comparative-10, chatgpt_comparative-20等）
                        for sub_metric_name, sub_values in values.items():
                            if (
                                isinstance(sub_values, dict)
                                and "error" not in sub_values
                            ):
                                for stat_name, value in sub_values.items():
                                    # comparison_resultsなどの詳細情報は保存しない
                                    if stat_name not in [
                                        "comparison_results",
                                        "individual_scores",
                                    ]:
                                        rows.append(
                                            {
                                                "file": file_name,
                                                "metric": sub_metric_name,  # chatgpt_comparative-10等, label-10等
                                                "statistic": stat_name,
                                                "value": convert_numpy_types(value),
                                            }
                                        )
                    else:
                        # 従来の単一結果の処理
                        for stat_name, value in values.items():
                            rows.append(
                                {
                                    "file": file_name,
                                    "metric": metric_name,
                                    "statistic": stat_name,
                                    "value": convert_numpy_types(value),
                                }
                            )

        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"評価結果を保存しました: {output_path}")

    else:
        print(f"未サポートの出力形式: {format}")


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """メイン関数"""
    print("評価設定:")
    print(OmegaConf.to_yaml(cfg))

    # 利用可能な指標を取得
    available_metrics = get_available_metrics()

    # 使用する指標を初期化
    metrics = []
    for metric_name in cfg.metrics:
        if metric_name in available_metrics:
            if metric_name == "length":
                metrics.append(
                    available_metrics[metric_name](min_length=cfg.min_length)
                )
            elif metric_name == "personaclr":
                metrics.append(
                    available_metrics[metric_name](max_reference=cfg.max_reference)
                )
            elif metric_name == "lunon_similar":
                if cfg.get("persona_model") is None or cfg.get("similar_model") is None:
                    print(
                        "警告: LUNON Similarを使用するには persona_model と similar_model を指定してください。スキップされます。"
                    )
                    continue
                metrics.append(
                    available_metrics[metric_name](
                        base_model_name=cfg.similar_model,
                        ft_model_name=cfg.persona_model,
                    )
                )
            elif metric_name == "nll":
                if cfg.get("persona_model") is None:
                    print(
                        "警告: NLLを使用するには persona_model を指定してください。スキップされます。"
                    )
                    continue
                metrics.append(
                    available_metrics[metric_name](ft_model_name=cfg.persona_model)
                )
            elif metric_name == "base_nll":
                metrics.append(
                    available_metrics[metric_name](base_model_name=cfg.base_model)
                )
            elif metric_name == "ipsi":
                if cfg.get("persona_model") is None:
                    print(
                        "警告: IPSIを使用するには persona_model を指定してください。スキップされます。"
                    )
                    continue
                metrics.append(
                    available_metrics[metric_name](
                        lambda_=cfg.ipsi_lambda,
                        base_model_name=cfg.base_model,
                        ft_model_name=cfg.persona_model,
                    )
                )
            elif metric_name == "lunon":
                if cfg.get("persona_model") is None:
                    print(
                        "警告: LUNONを使用するには persona_model を指定してください。スキップされます。"
                    )
                    continue
                metrics.append(
                    available_metrics[metric_name](
                        base_model_name=cfg.base_model,
                        ft_model_name=cfg.persona_model,
                    )
                )
            elif metric_name == "lunon_poly":
                if cfg.get("persona_model") is None:
                    print(
                        "警告: LUNON-polyを使用するには persona_model を指定してください。スキップされます。"
                    )
                    continue
                metrics.append(
                    available_metrics[metric_name](
                        lambda_=cfg.get("lunon_poly_lambda", 0.563),
                        base_model_name=cfg.base_model,
                        ft_model_name=cfg.persona_model,
                    )
                )
            elif metric_name == "chatgpt":
                # ChatGPT指標の設定を取得
                chatgpt_config = cfg.get("chatgpt", {})
                # memberが設定されている場合はexamples.csvのパスを構築
                examples_csv_path = None
                if cfg.get("member"):
                    examples_csv_path = f"evaluation/{cfg.member}/examples.csv"

                metrics.append(
                    available_metrics[metric_name](
                        prompt_file=chatgpt_config.get(
                            "prompt_file", "src/prompt/chatgpt_evaluation_prompt.txt"
                        ),
                        model=chatgpt_config.get("model", "gpt-4o"),
                        temperature=chatgpt_config.get("temperature", 0.0),
                        max_tokens=chatgpt_config.get("max_tokens", 10),
                        n_examples=chatgpt_config.get("n_examples", 20),
                        api_key=chatgpt_config.get("openai_api_key", None),
                        examples_csv_path=examples_csv_path,
                        eval_counts=chatgpt_config.get("eval_counts", None),
                    )
                )
            elif metric_name == "chatgpt_comparative":
                # ChatGPT相対評価指標の設定を取得
                chatgpt_comparative_config = cfg.get("chatgpt_comparative", {})

                metrics.append(
                    available_metrics[metric_name](
                        prompt_file=chatgpt_comparative_config.get(
                            "prompt_file",
                            "src/prompt/chatgpt_comparative_evaluation_prompt.txt",
                        ),
                        model=chatgpt_comparative_config.get("model", "gpt-4o"),
                        temperature=chatgpt_comparative_config.get("temperature", 0.0),
                        max_tokens=chatgpt_comparative_config.get("max_tokens", 200),
                        n_examples=chatgpt_comparative_config.get("n_examples", 20),
                        n_pairs=chatgpt_comparative_config.get("n_pairs", 300),
                        api_key=chatgpt_comparative_config.get("openai_api_key", None),
                        random_seed=chatgpt_comparative_config.get("random_seed", 42),
                        example_counts=chatgpt_comparative_config.get(
                            "example_counts", None
                        ),
                        verbose=chatgpt_comparative_config.get("verbose", False),
                        show_prompts=chatgpt_comparative_config.get(
                            "show_prompts", False
                        ),
                        show_responses=chatgpt_comparative_config.get(
                            "show_responses", False
                        ),
                        save_comparisons=chatgpt_comparative_config.get(
                            "save_comparisons", False
                        ),
                        load_comparisons=chatgpt_comparative_config.get(
                            "load_comparisons", False
                        ),
                        comparison_cache_dir=chatgpt_comparative_config.get(
                            "comparison_cache_dir", "cache/comparisons"
                        ),
                    )
                )
            elif metric_name == "bertscore":
                # BERTScore指標の設定を取得
                bertscore_config = cfg.get("bertscore", {})
                metrics.append(
                    available_metrics[metric_name](
                        model_name=bertscore_config.get(
                            "model_name",
                            "nlp-waseda/roberta-large-japanese-with-auto-jumanpp",
                        ),
                        device=bertscore_config.get("device", None),
                        max_length=bertscore_config.get("max_length", 512),
                        eval_counts=bertscore_config.get("eval_counts", None),
                    )
                )
            elif metric_name == "bleu":
                # BLEU指標の設定を取得
                bleu_config = cfg.get("bleu", {})
                metrics.append(
                    available_metrics[metric_name](
                        tokenize=bleu_config.get("tokenize", "ja-mecab"),
                        eval_counts=bleu_config.get("eval_counts", None),
                    )
                )
            else:
                metrics.append(available_metrics[metric_name]())
        else:
            print(f"警告: 未サポートの指標 '{metric_name}' はスキップされます")

    if not metrics:
        print("有効な評価指標が指定されていません。デフォルトでBLEUを使用します。")
        bleu_config = cfg.get("bleu", {})
        metrics.append(
            BLEUMetric(
                tokenize=bleu_config.get("tokenize", "ja-mecab"),
                eval_counts=bleu_config.get("eval_counts", None),
            )
        )

    # ラベル設定の処理
    labels = None
    if cfg.get("label"):
        # 全てのメトリクスに同じラベルを適用
        labels = {metric.name: cfg.label for metric in metrics}

    # 評価の実行
    results = evaluate_outputs(
        cfg.train_data,
        cfg.get("eval_dir", None),
        metrics,
        cfg.format,
        cfg.get("eval_file", None),
        labels,
    )

    # 結果の表示
    if results:
        print("\n評価結果:")
        for file_name, file_results in results.items():
            print(f"\n{file_name}:")
            for metric_name, values in file_results.items():
                if metric_name == "individual_scores":
                    continue
                print(f"  {metric_name}:")
                if isinstance(values, dict) and "error" not in values:
                    for stat_name, value in values.items():
                        print(f"    {stat_name}: {value}")
                else:
                    print(f"    {values}")

        # 結果を保存
        save_results(
            results,
            cfg.eval_dir
            if cfg.get("eval_dir") is not None
            else os.path.dirname(cfg.eval_file)
            if cfg.get("eval_file") is not None
            else os.path.dirname(cfg.train_data),
            cfg.format,
        )
    else:
        print("評価する出力ファイルが見つかりませんでした。")


if __name__ == "__main__":
    main()
