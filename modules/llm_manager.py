from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from config import settings
import torch

class LLMManager:
    def load_llm(self, max_tokens=256):
        """
        Load the LLM model using HuggingFacePipeline.
        :return: An instance of HuggingFacePipeline with the loaded model.
        """
        # Load the tokenizer and model

        # Configure the model to use 16-bit precision for efficiency
        bnb_config = BitsAndBytesConfig(   
            load_in_4bit=True,  # Load the model in 4-bit precision
            bnb_4bit_quant_type="nf4",  # Use non-fine-tuned quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for better performance
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
            bnb_4bit_use_triton=True,  # Use Triton for faster computation  
        )   

        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            quantization_config=bnb_config,  # Apply the quantization configuration
            device_map="auto"
        )

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_tokens,  # Set the maximum length for generated text
            do_sample=True,  # Enable sampling for more diverse outputs 
            temperature=0.7,  # Set the temperature for sampling
            top_p=0.95,  # Set the top-p value for nucleus sampling
        )


        return HuggingFacePipeline(pipeline=pipe)