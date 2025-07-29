#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人間評価とシステム評価の手法間でスピアマンの順位相関係数を計算するスクリプト
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_human_evaluation(file_path):
    """
    人間評価データを読み込む
    """
    df = pd.read_csv(file_path)

    # question_numカラムが既に存在するのでそれを使用してソート
    df = df.sort_values("question_num").reset_index(drop=True)

    return df


def load_system_evaluation(file_path):
    """
    システム評価データを読み込む
    """
    df = pd.read_csv(file_path)

    # indexカラムをソート（質問の順番に対応）
    df = df.sort_values("index").reset_index(drop=True)

    return df


def calculate_correlations(human_df, system_df):
    """
    スピアマンの順位相関係数を計算する
    """
    # 人間評価のmean_scoreとweighted_scoreを取得
    human_mean_ratings = human_df["mean_score"].values
    # human_weighted_ratings = human_df["weighted_score"].values

    # システム評価の各手法のカラムを特定
    exclude_columns = {"index", "prefix", "continuation"}
    system_methods = [col for col in system_df.columns if col not in exclude_columns]

    correlations = {}

    for method in system_methods:
        # 文字列として保存された辞書形式のデータを処理
        if system_df[method].dtype == 'object':
            try:
                # 辞書形式の文字列の場合、数値に変換を試みる
                import ast
                system_scores = []
                for val in system_df[method].values:
                    if isinstance(val, str) and val.startswith('{'):
                        # 辞書形式の場合、最初の値を取得
                        dict_val = ast.literal_eval(val)
                        first_key = list(dict_val.keys())[0]
                        system_scores.append(dict_val[first_key])
                    else:
                        system_scores.append(float(val))
                system_scores = np.array(system_scores)
            except:
                # 変換に失敗した場合はスキップ
                print(f"警告: {method}カラムのデータ変換に失敗しました。スキップします。")
                continue
        else:
            system_scores = system_df[method].values

        # mean_scoreとの相関を計算
        valid_indices_mean = ~(pd.isna(human_mean_ratings) | pd.isna(system_scores))
        if valid_indices_mean.sum() > 1:  # 最低2つのペアが必要
            human_valid = human_mean_ratings[valid_indices_mean]
            system_valid = system_scores[valid_indices_mean]

            # スピアマン相関係数を計算
            correlation, p_value = spearmanr(human_valid, system_valid)

            correlations[f"{method}_mean"] = {
                "spearman_correlation": correlation,
                "p_value": p_value,
                "n_samples": valid_indices_mean.sum(),
                "human_mean": human_valid.mean(),
                "human_std": human_valid.std(),
                "system_mean": system_valid.mean(),
                "system_std": system_valid.std(),
            }
        else:
            correlations[f"{method}_mean"] = {
                "spearman_correlation": np.nan,
                "p_value": np.nan,
                "n_samples": 0,
                "human_mean": np.nan,
                "human_std": np.nan,
                "system_mean": np.nan,
                "system_std": np.nan,
            }

        # weighted_scoreとの相関を計算
        # valid_indices_weighted = ~(
        #     pd.isna(human_weighted_ratings) | pd.isna(system_scores)
        # )
        # if valid_indices_weighted.sum() > 1:  # 最低2つのペアが必要
        #     human_valid = human_weighted_ratings[valid_indices_weighted]
        #     system_valid = system_scores[valid_indices_weighted]

        #     # スピアマン相関係数を計算
        #     correlation, p_value = spearmanr(human_valid, system_valid)

        #     correlations[f"{method}_weighted"] = {
        #         "spearman_correlation": correlation,
        #         "p_value": p_value,
        #         "n_samples": valid_indices_weighted.sum(),
        #         "human_mean": human_valid.mean(),
        #         "human_std": human_valid.std(),
        #         "system_mean": system_valid.mean(),
        #         "system_std": system_valid.std(),
        #     }
        # else:
        #     correlations[f"{method}_weighted"] = {
        #         "spearman_correlation": np.nan,
        #         "p_value": np.nan,
        #         "n_samples": 0,
        #         "human_mean": np.nan,
        #         "human_std": np.nan,
        #         "system_mean": np.nan,
        #         "system_std": np.nan,
        #     }

    return correlations


def save_correlation_results(correlations, output_file):
    """
    相関係数の結果をCSVファイルに保存する
    """
    results_df = pd.DataFrame.from_dict(correlations, orient="index")

    # カラムの順序を調整
    column_order = [
        "spearman_correlation",
        "p_value",
        "n_samples",
        "human_mean",
        "human_std",
        "system_mean",
        "system_std",
    ]
    results_df = results_df[column_order]

    # 相関係数でソート（降順）
    results_df = results_df.sort_values("spearman_correlation", ascending=False)

    results_df.to_csv(output_file, index=True, encoding="utf-8")

    return results_df


def print_summary(correlations_df):
    """
    結果のサマリーを表示する
    """
    print("\n=== スピアマンの順位相関係数 結果サマリー ===")

    # mean_scoreとweighted_scoreの結果を分離
    mean_methods = correlations_df[correlations_df.index.str.endswith("_mean")]
    weighted_methods = correlations_df[correlations_df.index.str.endswith("_weighted")]

    print(f"評価手法数: {len(mean_methods)}")
    print(
        f"サンプル数: {correlations_df['n_samples'].iloc[0] if not correlations_df.empty else 0}"
    )

    # mean_scoreとの相関を表示
    print("\n=== mean_score (単純平均) との相関 ===")
    mean_valid = mean_methods.dropna(subset=["spearman_correlation"])
    if not mean_valid.empty:
        mean_sorted = mean_valid.sort_values("spearman_correlation", ascending=False)
        for method, row in mean_sorted.iterrows():
            method_name = method.replace("_mean", "")
            significance = (
                "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            )
            print(
                f"  {method_name}: {row['spearman_correlation']:.4f} (p={row['p_value']:.4f}){significance}"
            )

        best_mean = mean_sorted.index[0].replace("_mean", "")
        worst_mean = mean_sorted.index[-1].replace("_mean", "")
        print(
            f"最高: {best_mean} (r={mean_sorted.iloc[0]['spearman_correlation']:.4f})"
        )
        print(
            f"最低: {worst_mean} (r={mean_sorted.iloc[-1]['spearman_correlation']:.4f})"
        )

    # weighted_scoreとの相関を表示
    # print("\n=== weighted_score (加重平均) との相関 ===")
    # weighted_valid = weighted_methods.dropna(subset=["spearman_correlation"])
    # if not weighted_valid.empty:
    #     weighted_sorted = weighted_valid.sort_values(
    #         "spearman_correlation", ascending=False
    #     )
    #     for method, row in weighted_sorted.iterrows():
    #         method_name = method.replace("_weighted", "")
    #         significance = (
    #             "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
    #         )
    #         print(
    #             f"  {method_name}: {row['spearman_correlation']:.4f} (p={row['p_value']:.4f}){significance}"
    #         )

    #     best_weighted = weighted_sorted.index[0].replace("_weighted", "")
    #     worst_weighted = weighted_sorted.index[-1].replace("_weighted", "")
    #     print(
    #         f"最高: {best_weighted} (r={weighted_sorted.iloc[0]['spearman_correlation']:.4f})"
    #     )
    #     print(
    #         f"最低: {worst_weighted} (r={weighted_sorted.iloc[-1]['spearman_correlation']:.4f})"
    #     )

    # print("\n** p < 0.01, * p < 0.05")


def main():
    """
    メイン関数
    """
    if len(sys.argv) < 3:
        print(
            "使用法: python calculate_spearman_correlation.py <human_evaluation_file> <system_evaluation_file> [output_file]"
        )
        print(
            "例: python calculate_spearman_correlation.py evaluation/hiyori/濱岸ひよりらしさの評価_question_summary.csv evaluation/hiyori/eval/for_evaluation_individual_scores.csv"
        )
        sys.exit(1)

    human_file = sys.argv[1]
    system_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        # データ読み込み
        print(f"人間評価データ読み込み: {human_file}")
        human_df = load_human_evaluation(human_file)
        print(f"質問数: {len(human_df)}")

        print(f"システム評価データ読み込み: {system_file}")
        system_df = load_system_evaluation(system_file)
        print(f"評価対象数: {len(system_df)}")

        # データ数の整合性チェック
        if len(human_df) != len(system_df):
            print(
                f"警告: データ数が一致しません (人間評価: {len(human_df)}, システム評価: {len(system_df)})"
            )
            min_len = min(len(human_df), len(system_df))
            human_df = human_df.head(min_len)
            system_df = system_df.head(min_len)
            print(f"最初の{min_len}件で分析を実行します")

        # 相関係数計算
        correlations = calculate_correlations(human_df, system_df)

        # 出力ファイル名決定
        if output_file is None:
            system_path = Path(system_file)
            output_file = (
                system_path.parent / f"spearman_correlations_{system_path.stem}.csv"
            )

        # 結果保存
        correlations_df = save_correlation_results(correlations, output_file)

        print(f"\n結果保存: {output_file}")

        # サマリー表示
        print_summary(correlations_df)

        print("\n=== 使用例 ===")
        print("# 結果を読み込み")
        print("import pandas as pd")
        print(f"results = pd.read_csv('{output_file}', index_col=0)")
        print("print(results)")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
