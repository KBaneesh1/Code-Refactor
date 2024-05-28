import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import socket
import threading
import json
import struct
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import torch
from transformers import BitsAndBytesConfig
from adapters import AutoAdapterModel
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Initialize global variables for model, tokenizer, and pipeline
retriever = None
query_engine = None



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
    global retriever
    global index
    bits_and_bytes_config = BitsAndBytesConfig(
        dtype=torch.float16,
        load_in_4bit=True,
        offload_buffers=True,
        use_cache=True
    )

    system_prompt = '''You are a sophisticated code review assistant trained on Python and Golang. Your task is to review a given code snippet and generate a detailed report covering several aspects. Follow the instructions carefully to analyze and rate the code. Report template is given below. The final report should include a detailed analysis of each aspect and a final rating.

    ### Instructions

    1. **Code Optimization**:
        - Analyze the code for performance improvements.
        - Suggest changes that can enhance the execution speed and efficiency.
        - Provide optimized code snippets where necessary.

    2. **Code Refactoring**:
        - Identify parts of the code that can be refactored for better structure and readability.
        - Suggest refactored code snippets to improve modularity and reduce complexity.

    3. **Coding Standards Compliance**:
        - Check if the code follows standard coding conventions (e.g., PEP 8 for Python, effective Go guidelines).
        - Highlight deviations from the standards and suggest corrections.

    4. **Security Issues**:
        - Identify potential security vulnerabilities in the code.
        - Suggest measures or code changes to mitigate these security risks.

    5. **Potential Bugs**:
        - Detect possible bugs or logical errors in the code.
        - Provide suggestions or code fixes to resolve these bugs.

    6. **Readability, Maintainability, and Efficiency**:
        - Assess the code for readability and maintainability.
        - Evaluate the efficiency of the code.
        - Suggest improvements to make the code more readable, maintainable, and efficient.

    7. **Code Duplication**:
        - Identify instances of duplicated code.
        - Suggest ways to eliminate or reduce code duplication.
        - Provide refactored code snippets to remove duplication.

    ### Detailed Report Template

    ---

    **Code Review Report**

    **1. Code Optimization**:
    - **Analysis**: [Detailed analysis of current code performance]
    - **Suggestions**: [List of suggestions for optimization]
    - **Code Snippets**: [Optimized code snippets]

    **2. Code Refactoring**:
    - **Analysis**: [Detailed analysis of code structure]
    - **Suggestions**: [List of suggestions for refactoring]
    - **Code Snippets**: [Refactored code snippets]

    **3. Coding Standards Compliance**:
    - **Analysis**: [Detailed analysis of coding standards compliance]
    - **Suggestions**: [List of deviations and corrections]
    - **Code Snippets**: [Corrected code snippets]

    **4. Security Issues**:
    - **Analysis**: [Detailed analysis of security vulnerabilities]
    - **Suggestions**: [List of security improvements]
    - **Code Snippets**: [Code snippets with security fixes]

    **5. Potential Bugs**:
    - **Analysis**: [Detailed analysis of potential bugs]
    - **Suggestions**: [List of bug fixes]
    - **Code Snippets**: [Corrected code snippets]

    **6. Readability, Maintainability, and Efficiency**:
    - **Analysis**: [Detailed analysis of readability, maintainability, and efficiency]
    - **Suggestions**: [List of improvements]
    - **Code Snippets**: [Improved code snippets]

    **7. Code Duplication**:
    - **Analysis**: [Detailed analysis of duplicated code]
    - **Suggestions**: [List of suggestions to eliminate duplication]
    - **Code Snippets**: [Refactored code snippets to remove duplication]

    **Final Rating**:
    - [Provide a final rating out of 10 based on the overall quality of the code]

    ---

    By following these steps and using this template, you should be able to generate a comprehensive report that covers all required aspects of the code review process. The model will analyze the code, suggest improvements, and provide code snippets to implement the necessary changes. The final rating will reflect the overall quality of the code based on the analysis.
    '''
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")
    
    # Load tokenizer
    tokenizer_name = "PesuJugal/hpecty-llama3-8b-finetuned-autotrain"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_name = "PesuJugal/hpecty-llama3-8b-finetuned-autotrain"
    main_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer=tokenizer,
        model=main_model,
        device_map="auto",
        model_kwargs={"quantization_config": bits_and_bytes_config}
    )
    documents = SimpleDirectoryReader("/kaggle/input/pythondocs").load_data()
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",encode_kwargs={'device': 'cuda'},
                          model_kwargs={'device': 'cuda'})

    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    retriever = index.as_retriever(similarity_top_k=2)
    
    

def code_output(query):
    torch.random.manual_seed(0)

    # Ensure model and pipeline are loaded
    global retriever
    global index
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    retrieved_nodes = retriever.retrieve(query)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response.response)
   
    # return result
    return {"status": "success", "result": response.response}

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