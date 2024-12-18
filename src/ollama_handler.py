import ollama


def generate_reasoning(sentence, model: str):
    """Generate reasoning chains using the Ollama Mistral model."""
    response = ollama.generate(
        model=model,
        prompt=f"Generate three reasoning steps for this statement: {sentence}",
    )
    return response.get("response", "")


def generate_response_from_prompt(prompt, model: str):
    """Generate the meaning for a sentence using a given Ollama model."""
    response = ollama.generate(model=model, prompt=prompt)
    return response.get("response", "")
