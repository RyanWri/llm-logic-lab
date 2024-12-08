import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


print(f"MPS Available: {torch.backends.mps.is_available()}")


def load_model(model_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a text-generation pipeline with the correct device
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )
    return generator


# Example prompts
prompts = [
    "Generate three reasoning steps for this statement: John couldn't find his glasses while they were on his head.",
    "Event: PersonX shares their food. Dimension: xIntent.",
    "Event: PersonX walks their dog. Dimension: oEffect.",
]

model_path = "/home/linuxu/models-logs/distilgpt2-fine-tuned/"
model_name = "distilbert/distilgpt2"
generator = load_model(model_path)
# Generate responses
for prompt in prompts:
    response = generator(prompt, max_length=50, truncation=True, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[0]['generated_text']}\n")
