from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from transformers import PreTrainedTokenizer
from peft import LoraConfig, get_peft_model
from typing import Tuple, Any, Dict

class PersonaModel:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit

    def setup_model(self) -> Tuple[Any, Any]:
        """モデルとトークナイザーをセットアップします"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.use_4bit
        )
        
        # モデルをGPUに移動
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.use_4bit:
            model = model.to(device)
        # model = model.to(device)
        print(f"Model loaded on: {device}")
        
        return model, tokenizer
    
    def apply_lora(self, model: Any, lora_config: Dict[str, Any]) -> Any:
        """LoRAを適用します"""
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_rslora=lora_config["use_rslora"]
        )
        
        print("LoRA設定を適用しました")
        return model 