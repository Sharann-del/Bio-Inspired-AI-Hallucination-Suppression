import requests
import json

OLLAMA_HOST = "http://127.0.0.1:11434"
TEST_PROMPT = "Who won the Nobel Prize in Physics in 1921 and why?"

def test_model(model_name):
    print(f"\nTesting {model_name}...")
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model_name,
            "prompt": TEST_PROMPT,
            "stream": False
        }
    )
    result = response.json()
    print(f"Response: {result['response'][:300]}")
    print(f"Done tokens: {result['eval_count']}")
    return result

if __name__ == "__main__":
    print("=== Inference Test ===")
    print(f"Prompt: {TEST_PROMPT}\n")
    test_model("mistral")
    test_model("llama2:7b")
    print("\n=== Both models working ===")
