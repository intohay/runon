from typing import List, Dict, Any, Optional
from src.model.openai_model import OpenAIModel

class StyleMimicker:
    def __init__(self, openai_model: Optional[OpenAIModel] = None):
        """
        few-shotアプローチで特定の人物の発話スタイルを模倣するシンプルなクラス
        
        Parameters:
            openai_model: OpenAIモデルのインスタンス。Noneの場合は新たに作成します。
        """
        self.openai_model = openai_model or OpenAIModel()
        self.examples = []
        
    def add_examples(self, examples: List[str]):
        """
        模倣するための発話例を追加します
        
        Parameters:
            examples: 発話例のリスト
        """
        self.examples.extend([ex.strip() for ex in examples if ex.strip()])
    
    def generate_mimicking_text(
        self, 
        prompt: str, 
        system_instruction: str = "以下の発話例の文体や話し方を模倣して回答してください。", 
        model: str = "gpt-4o", 
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        few-shotアプローチで発話スタイルを模倣したテキストを生成します
        
        Parameters:
            prompt: 生成の元となるプロンプト（ユーザーからの質問や指示）
            system_instruction: システム指示
            model: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性（高いほど多様）
            
        Returns:
            生成されたテキスト
        """
        if not self.examples:
            raise ValueError("発話例が追加されていません。add_examples()で発話例を追加してください。")
        
        # 発話例をシステムプロンプトに組み込む
        examples_text = "\n".join([f"- {ex}" for ex in self.examples])
        system_prompt = f"{system_instruction}\n\n発話例:\n{examples_text}"
        
        # 生成
        response = self.openai_model.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_with_persona(
        self,
        prompt: str,
        persona_description: str = "",
        model: str = "gpt-4o",
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        キャラクター設定と発話例を組み合わせて生成します
        
        Parameters:
            prompt: 生成の元となるプロンプト
            persona_description: キャラクター設定の説明
            model: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性
            
        Returns:
            生成されたテキスト
        """
        if not self.examples:
            raise ValueError("発話例が追加されていません。add_examples()で発話例を追加してください。")
        
        # 発話例をフォーマット
        examples_text = "\n".join([f"- {ex}" for ex in self.examples])
        
        # システムプロンプト
        system_prompt = f"""あなたは以下の特徴を持つキャラクターとして振る舞ってください。

{persona_description}

以下は実際の発言例です：
{examples_text}

上記の発言例を参考に、同じ話し方や言葉遣い、文体で応答してください。
キャラクターになりきって一人称で返答してください。"""
        
        # 生成
        response = self.openai_model.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        prompt: str,
        context: str,
        model: str = "gpt-4o",
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        コンテキスト情報と発話例を組み合わせて生成します
        
        Parameters:
            prompt: 生成の元となるプロンプト
            context: 状況や背景などのコンテキスト情報
            model: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性
            
        Returns:
            生成されたテキスト
        """
        if not self.examples:
            raise ValueError("発話例が追加されていません。add_examples()で発話例を追加してください。")
        
        # 発話例をフォーマット
        examples_text = "\n".join([f"- {ex}" for ex in self.examples])
        
        # システムプロンプト
        system_prompt = f"""以下の文体や話し方を模倣して回答してください。

発話例:
{examples_text}

コンテキスト情報:
{context}

上記の発話例の文体を忠実に模倣し、コンテキスト情報を考慮して回答してください。"""
        
        # 生成
        response = self.openai_model.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_with_qa_examples(
        self,
        prompt: str,
        qa_examples: List[Dict[str, str]],
        model: str = "gpt-4o",
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        質問と回答のペアを使ったfew-shot生成を行います
        
        Parameters:
            prompt: 生成の元となるプロンプト
            qa_examples: 質問と回答のペア [{"question": "質問文", "answer": "回答文"}]
            model: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性
            
        Returns:
            生成されたテキスト
        """
        if not qa_examples:
            raise ValueError("質問と回答のペアが指定されていません")
        
        # few-shotメッセージを作成
        messages = [{"role": "system", "content": "以下の回答例の文体や話し方を模倣して回答してください。"}]
        
        for ex in qa_examples:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["answer"]})
        
        # ユーザーの質問を追加
        messages.append({"role": "user", "content": prompt})
        
        # 生成
        response = self.openai_model.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content 