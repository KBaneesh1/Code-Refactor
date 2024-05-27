import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import socket
import threading
import json
import struct

# Initialize global variables for model, tokenizer, and pipeline
model = None
tokenizer = None
pipe = None


def recvall(sock):
    """Helper function to receive the full message"""
    raw_msglen = recvall_part(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall_part(sock, msglen)

def recvall_part(sock, n):
    """Helper function to receive n bytes or return None if EOF is hit"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def load_model():
    global model, tokenizer, pipe
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
    # model = 1
    # tokenizer = 2
    # pipe = 3

def code_output(query):
    torch.random.manual_seed(0)

    # Ensure model and pipeline are loaded
    global model, tokenizer, pipe
    print(model,tokenizer,pipe)
    if model is None or pipe is None:
        load_model()

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
    result = output[0]['generated_text']
    # print(model,tokenizer,pipe)
    # refactored_code = "123 lets go"
    torch.cuda.empty_cache()
    # return result
    return {"status": "success", "result": result}

def handle_client(client_socket):
    try:
        data = recvall(client_socket)
        if not data:
            return
        message = json.loads(data.decode('utf-8'))
        print(f"Received message: {message}")
        response = code_output(message['input_data'])
        response_data = json.dumps(response).encode('utf-8')
        response_len = struct.pack('>I', len(response_data))
        client_socket.sendall(response_len + response_data)
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()

def start_server():
    PORT = 8000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', PORT))
    server_socket.listen(10)  # Allow up to 10 connections
    IP = socket.gethostbyname(socket.gethostname())

    print(f"Server listening on port {PORT} and IP {IP} ")

    # Load the model when the server starts
    load_model()

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()   
    server_socket.close()

if __name__ == "__main__":
    start_server()