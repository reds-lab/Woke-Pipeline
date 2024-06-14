import gc
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# load_hugging_config()
# Set the default cache directory for Hugging Face models and datasets.
# please keep 0 away in this list as somehow it does not support muti-gpu
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], 'hub')
os.environ["HF_ASSETS_CACHE"] = os.path.join(os.environ["HF_HOME"], 'assets')
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')
from transformers import AutoTokenizer
from .utils.prompt_util import find_prompt_template
import time
import torch
from tqdm import tqdm
import json
from .utils.models_util import load_model_and_tokenizer
from .utils.config_util import load_models_dict_json,load_api_models,load_local_models_dict_json
from vllm import LLM, SamplingParams
from .utils.claude_generation import get_response_claude
from .utils.gemini_generation import get_response_gemini
from .utils.gpt_generation import get_response_gpt
model_dict = load_models_dict_json()
api_models_list = load_api_models()
local_model_dict = load_local_models_dict_json()


def feed_forward(model_name, prompts):

    responses = []
    model_name = model_name.lower().strip()
    model_id = model_name
    if model_name in model_dict.keys() or model_name in local_model_dict.keys():
        model_id = model_dict.get(model_name) or local_model_dict.get(model_name)

    
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        

        vllm_model = LLM(model=model_id,
                        trust_remote_code=True,
                        tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
                        #  disable_custom_all_reduce=True,
                        gpu_memory_utilization=0.9,
                        dtype='bfloat16',
                        swap_space=8,
                        )

        dialogs = find_prompt_template(prompts, model_name, tokenizer)
    
        sampling_params = SamplingParams(temperature=0, max_tokens=256)

        vllm_outputs = vllm_model.generate(dialogs, sampling_params)
    
        responses = [output.outputs[0].text.strip() for output in vllm_outputs]
        
        del vllm_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    elif model_name in api_models_list:
        print(f"API Model: {model_name}")
        for prompt in tqdm(prompts):
            while True:
                try:
                    if "claude" in model_name:
                        response = get_response_claude(prompt, model_name)
                    elif "gpt" in model_name:
                        response = get_response_gpt(prompt, model_name)
                    elif "gemini" in model_name:
                        response = get_response_gemini(prompt, model_name)
                    
                    responses.append(response)
                    # save_response(model_name, model_id, prompt, response, output_dir)
                    break
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    print("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
        
        return responses
    else:
        raise ValueError("Model Name not supported/included")
    
    qa_pairs = zip(prompts, responses)
    
    return qa_pairs
    
