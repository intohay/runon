from typing import Any, Dict

from datasets import load_dataset


class PersonaDataset:
    def __init__(
        self,
        train_file: str,
        eval_file: str,
        tokenizer: Any,
        max_length: int = 2048,
        debug_mode: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_file = train_file
        self.eval_file = eval_file
        self.debug_mode = debug_mode

    def load_dataset(self) -> Dict[str, Any]:
        """データセットをロードします"""
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.train_file,
                "validation": self.eval_file,
            },
            download_mode="force_redownload",
        )
        return dataset

    def apply_chat_template(self, example):
        """チャットテンプレートを適用します（chat_templateがない場合はmessagesを連結）"""
        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            # chat_templateがある場合はテンプレートを適用
            example["text"] = self.tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        else:
            # chat_templateがない場合はmessagesを単純に連結

            example["text"] = (
                self.tokenizer.bos_token
                + "\n".join(
                    [
                        m["content"]
                        for m in example["messages"]
                        if isinstance(m, dict) and m.get("role") == "assistant"
                    ]
                )
                + self.tokenizer.eos_token
            )

        return example

    def prepare_dataset(self) -> Dict[str, Any]:
        """データセットを前処理します"""
        dataset = self.load_dataset()

        # チャットテンプレートを適用
        dataset = dataset.map(
            self.apply_chat_template,
            remove_columns=["messages"],  # messagesカラムは不要になる
        )

        # デバッグモードでデータセットのサイズを制限
        if self.debug_mode:
            print("\nデバッグモード：データセットのサイズを制限します")
            train_size = min(100, len(dataset["train"]))
            val_size = min(20, len(dataset["validation"]))
            dataset["train"] = dataset["train"].select(range(train_size))
            dataset["validation"] = dataset["validation"].select(range(val_size))
            print(f"学習データ: {train_size}件, 検証データ: {val_size}件")
            # デバッグ用：最初のデータを表示
            print("\n最初の学習データの例:")
            print(dataset["train"][0]["text"])

        return dataset
