import os
import openai
from typing import List, Dict, Any, Optional, Union
import json
import time

class OpenAIModel:
    def __init__(
        self,
        api_key: Optional[str] = None, 
        organization: Optional[str] = None
    ):
        """
        OpenAIのAPIを使用するためのクラス
        
        Parameters:
            api_key: OpenAIのAPIキー。Noneの場合は環境変数から取得
            organization: OpenAIのOrganization ID。Noneの場合は環境変数から取得
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        
        if not self.api_key:
            raise ValueError("APIキーが設定されていません。APIキーを引数で渡すか、OPENAI_API_KEY環境変数を設定してください。")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
    
    def generate(
        self, 
        prompt: str, 
        model: str = "gpt-4o", 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> str:
        """
        プロンプトに基づいてテキストを生成します
        
        Parameters:
            prompt: 生成の元となるプロンプト
            model: 使用するモデル
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性（高いほど多様）
            top_p: 核サンプリングの確率閾値
            frequency_penalty: 単語の繰り返しに対するペナルティ
            presence_penalty: トピックの繰り返しに対するペナルティ
            stop: 生成を停止する文字列または文字列リスト
            
        Returns:
            生成されたテキスト
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        
        return response.choices[0].message.content
    
    def prepare_finetune_data(
        self, 
        data: List[Dict[str, str]], 
        output_file: str = "finetune_data.jsonl"
    ) -> str:
        """
        ファインチューニング用のデータを準備します
        
        Parameters:
            data: ファインチューニング用のデータ。各要素は{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]} の形式
            output_file: 出力ファイル名
            
        Returns:
            出力ファイルのパス
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"ファインチューニングデータを {output_file} に保存しました")
        return output_file
    
    def create_finetune(
        self, 
        training_file: str,
        model: str = "gpt-3.5-turbo",
        suffix: Optional[str] = None,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        ファインチューニングジョブを作成します
        
        Parameters:
            training_file: トレーニングデータのファイルID
            model: ベースモデル
            suffix: モデル名の接尾辞
            validation_file: 検証データのファイルID
            hyperparameters: ハイパーパラメータ
            
        Returns:
            ファインチューニングジョブのID
        """
        # ファイルをアップロード
        if not training_file.startswith("file-"):
            file = self.client.files.create(
                file=open(training_file, "rb"),
                purpose="fine-tune"
            )
            training_file = file.id
            print(f"トレーニングファイルをアップロードしました: {training_file}")
        
        if validation_file and not validation_file.startswith("file-"):
            file = self.client.files.create(
                file=open(validation_file, "rb"),
                purpose="fine-tune"
            )
            validation_file = file.id
            print(f"検証ファイルをアップロードしました: {validation_file}")
        
        # ファインチューニングジョブを作成
        job_params = {
            "training_file": training_file,
            "model": model,
        }
        
        if suffix:
            job_params["suffix"] = suffix
        
        if validation_file:
            job_params["validation_file"] = validation_file
            
        if hyperparameters:
            job_params["hyperparameters"] = hyperparameters
        
        # ファインチューニングジョブを作成
        job = self.client.fine_tuning.jobs.create(**job_params)
        
        print(f"ファインチューニングジョブを作成しました: {job.id}")
        return job.id
    
    def get_finetune_status(self, job_id: str) -> Dict[str, Any]:
        """
        ファインチューニングジョブのステータスを取得します
        
        Parameters:
            job_id: ファインチューニングジョブのID
            
        Returns:
            ファインチューニングジョブの情報
        """
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return job
    
    def wait_for_finetune_completion(
        self, 
        job_id: str, 
        poll_interval: int = 60
    ) -> Dict[str, Any]:
        """
        ファインチューニングジョブの完了を待ちます
        
        Parameters:
            job_id: ファインチューニングジョブのID
            poll_interval: ステータス確認の間隔（秒）
            
        Returns:
            完了したファインチューニングジョブの情報
        """
        while True:
            job = self.get_finetune_status(job_id)
            status = job.status
            
            if status == "succeeded":
                print("ファインチューニングが完了しました！")
                return job
            
            if status == "failed":
                raise Exception(f"ファインチューニングが失敗しました: {job}")
            
            print(f"ファインチューニング進行中... (ステータス: {status})")
            time.sleep(poll_interval)
    
    def generate_with_finetuned_model(
        self, 
        prompt: str, 
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> str:
        """
        ファインチューニングしたモデルを使用してテキストを生成します
        
        Parameters:
            prompt: 生成の元となるプロンプト
            model: 使用するファインチューニング済みモデル
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性（高いほど多様）
            top_p: 核サンプリングの確率閾値
            frequency_penalty: 単語の繰り返しに対するペナルティ
            presence_penalty: トピックの繰り返しに対するペナルティ
            stop: 生成を停止する文字列または文字列リスト
            
        Returns:
            生成されたテキスト
        """
        return self.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        ) 