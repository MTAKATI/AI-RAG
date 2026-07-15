# Let's import these libraries
import torch  # PyTorch, the backend for transformers
from transformers import AutoModelForCausalLM
from transformers import TextStreamer, TextIteratorStreamer
from transformers import AutoTokenizer
from threading import Thread
from pdf_reader import PdfReader
from local_embedding import LocalEmbedding
import os
from huggingface_hub import login


class AiModel():

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        '''
            initializing my AiModel class where we need the model name to create a tokenizer and a model
            the tokenizer will transform our text into numbers for our model to understand then will transform the numbers from the model to text so we understand it
            the model is the LLM that will think and give us the answers to our questions
        '''
        self.model_name = model_name
        print("running checks to make sure everything is good...")
        self.hugging_face_auth()
        self.hardware_check()
        print("we are creating the model this might take a while please wait...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_name, torch_dtype="auto", device_map="auto")
    

    def hardware_check(self):
        '''
            making sure we are working on a local GPU rather than CPU to take advantage of Local LLMs
        '''
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: No GPU detected.")
    

    def hugging_face_auth(self):
        '''
            in order to download the right model to work on some of the model are gated by HuggingFace therefore we must authenticate first
        '''
        # getting token from .env
        HUGGING_FACE_TOKEN=os.environ.get("HF_TOKEN")

        # logging in
        print("Attempting Hugging Face login...")
        login(token=HUGGING_FACE_TOKEN)
        print("Login successful!")


    def ask_a_question(self, prompt="Hello there!"):
        '''
            formats the prompt as a chat message so the instruction-tuned model
            knows to respond rather than do raw text completion
        '''
        # wrap the prompt in the chat format the model was trained on
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # getting input and prompt
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # Streaming output — tokens are printed to the terminal as they are generated,
        # instead of waiting for the full response to be built in memory first.
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # max_new_tokens caps the response length; streamer handles all printing internally.
        self.model.generate(**inputs, max_new_tokens=1000, streamer=streamer)
    
    def ask_a_question_from_pdf(self, pdf_path, prompt="tell me what is this pdf about"):
        '''
        Now cleanly separates System Instructions from the User query + Context
        '''
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()
        
        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, 10)

        # Structure messages correctly so the Chat Template can do its job
        messages = [
            {
                "role": "system", 
                "content": "You are an AI assistant. Answer the following question based *only* on the provided document text. If the answer is not found in the document, say 'The document does not contain information on this topic.' Do not use any prior knowledge."
            },
            {
                "role": "user", 
                "content": f"Document Text:\n---\n{relevant_sections}\n---\n\nQuestion: {prompt}"
            }
        ]
        
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.generate(**inputs, max_new_tokens=1000, streamer=streamer)


    def ask_a_question_from_pdf_stream(self, pdf_path: str, prompt: str = "tell me what is this pdf about", local_embedding=None):
        '''
        Clean, streaming-friendly implementation preventing double-templating.
        '''
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()
            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, k=10)
        
        # Structure messages natively here too
        messages = [
            {
                "role": "system", 
                "content": "You are an AI assistant. Answer the following question based *only* on the provided document text. If the answer is not found in the document, say 'The document does not contain information on this topic.' Do not use any prior knowledge."
            },
            {
                "role": "user", 
                "content": f"Document Text:\n---\n{relevant_sections}\n---\n\nQuestion: {prompt}"
            }
        ]
        
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30.0
        )

        thread = Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, max_new_tokens=1000, streamer=streamer),
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()


    def full_prompt_for_rag(self, relevent_sections, question_prompt):
        '''
            this is a prompt constructor that will put together the user question, the pdf relevant sections, and system prompt
        '''
        return f"""
            <|system|>
                You are an AI assistant. Answer the following question based *only* on the provided document text. 
                If the answer is not found in the document, say "The document does not contain information on this topic." Do not use any prior knowledge.

                Document Text:
                ---
                    {relevent_sections}
                ---
            <|end|>
            |user|>
                Question: {question_prompt}
            <|end|>
            <|assistant|>
                Answer:
    """


#new_ai_model = AiModel()
#new_ai_model.ask_a_question_from_pdf("./pdfs/2025-q1-earnings-transcript.pdf")