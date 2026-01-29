#pip install transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
text = "¿Cual es la capital de españa?"
tokens = tokenizer.tokenize(text)
print(len(tokens))  # Número exacto de tokens