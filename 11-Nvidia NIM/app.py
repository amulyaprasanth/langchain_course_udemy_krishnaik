import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
nv_api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = str(nv_api_key)
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"write an essay on machine learning"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)

