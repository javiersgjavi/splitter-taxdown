from openai import OpenAI

class Splitter:
    def __init__(self, base_model, api_key, base_prompt, seed=42):
        self.base_model = base_model
        self.api_key = api_key
        self.base_prompt = base_prompt
        self.seed = seed

    def get_response(self, custom_prompt, temperature=1.0):
        client = OpenAI(api_key=self.api_key)
        prompt = f'{self.base_prompt}\n{custom_prompt}'
        answer = client.chat.completions.create(
            model=self.base_model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            seed=self.seed,
            temperature=temperature
        )
        return answer.choices[0].message.content
    

class EmbeddingGenerator:
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding