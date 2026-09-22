import os
from transformers import AutoTokenizer

def main():
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=token)
    
    for digit in ["0", "1", "2", "3"]:
        tokens = tokenizer.encode(digit, add_special_tokens=False)
        print(f"Digit '{digit}' -> Tokens: {tokens}")
        assert len(tokens) == 1, f"Digit {digit} is not a single token!"
        
    prompt = "Score:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt '{prompt}' -> Tokens: {prompt_tokens}")
    
    prompt_with_digit = "Score:0"
    combined_tokens = tokenizer.encode(prompt_with_digit, add_special_tokens=False)
    print(f"Combined '{prompt_with_digit}' -> Tokens: {combined_tokens}")
    
    assert combined_tokens == prompt_tokens + tokenizer.encode("0", add_special_tokens=False), "Prompt does not end cleanly before digit!"
    print("Verification passed!")

if __name__ == "__main__":
    main()
