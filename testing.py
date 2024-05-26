import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def code_output(query):
  torch.random.manual_seed(0)

  # Load the model and tokenizer
  global model
  global tokenizer
  global pipe

  messages = [
    {"role": "user", "content": "Can you review the following code and provide ratings and suggestions for improvement?"},
    {"role": "user", "content": query}
  ]


  seq_len = 10 * len(query)
  generation_args = {
    "max_new_tokens": seq_len,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
  }
  # Generate the review
  output = pipe(messages, **generation_args)
  print(output[0]['generated_text'])
  torch.cuda.empty_cache()
  return output[0]['generated_text']

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
  )
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def main():
    selected_text = sys.argv[1]
    # Process the selected text as needed
    print(selected_text)  # Example: Convert selected text to uppercase

if __name__ == "__main__":
    main()
