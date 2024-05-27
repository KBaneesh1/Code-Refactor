# import sys
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Initialize global variables for model, tokenizer, and pipeline
# model = None
# tokenizer = None
# pipe = None

# def load_model():
#     global model, tokenizer, pipe
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     "microsoft/Phi-3-mini-4k-instruct",
#     #     device_map="cuda",
#     #     torch_dtype="auto",
#     #     trust_remote_code=True
#     # )
#     # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
#     # pipe = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     # )
#     model=1
#     tokenizer = 2
#     pipe = 3

# def code_output(query):
#     torch.random.manual_seed(0)

#     # Ensure model and pipeline are loaded
#     global model, tokenizer, pipe
#     print(model,tokenizer,pipe)
#     # if model is None or pipe is None:
#     #     load_model()

#     # messages = [
#     #     {"role": "user", "content": "Can you review the following code and provide ratings and suggestions for improvement?"},
#     #     {"role": "user", "content": query}
#     # ]

#     # seq_len = 10 * len(query)
#     # generation_args = {
#     #     "max_new_tokens": seq_len,
#     #     "return_full_text": False,
#     #     "temperature": 0.0,
#     #     "do_sample": False,
#     # }
#     # # Generate the review
#     # output = pipe(messages, **generation_args)
#     # result = output[0]['generated_text']
#     # print(result)
#     # torch.cuda.empty_cache()
#     # return result

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: script.py <function_name> <argument>")
#         sys.exit(1)

#     function_name = sys.argv[1]
#     argument = sys.argv[2]

#     if function_name == "load_model":
#         load_model()
#     elif function_name == "code_output":
#         result = code_output(argument)
#         print(result)
#     else:
#         print(f"Unknown function: {function_name}")
import sys
import json

class ModelHandler:
    def __init__(self):
        self.model = None

    def load_model(self):
        # Replace with actual model loading code
        self.model = "Loaded Model"
        # print("Model loaded", file=sys.stderr)

    def code_output(self, input_data):
        if self.model is None:
            raise Exception("Model not loaded. Please load the model first.")
        # Replace with actual code output logic
        output = f"Output for {input_data} using {self.model}"
        return output

model_handler = ModelHandler()

def handle_command(command, input_data):
    if command == "load_model":
        model_handler.load_model()
        return {"status": "success"}
    elif command == "code_output":
        try:
            result = model_handler.code_output(input_data)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "Unknown command"}

if __name__ == "__main__":
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            command = message["command"]
            input_data = message.get("input_data", "")
            response = handle_command(command, input_data)
        except Exception as e:
            response = {"status": "error", "message": str(e)}
        print(json.dumps(response))
        sys.stdout.flush()
