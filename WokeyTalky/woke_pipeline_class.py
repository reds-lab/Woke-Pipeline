# package_name/my_class.py
import os

from feedforward import feed_forward
from utils.eval_util import load_prompt_format

class WokePipeline:
    def __init__(self, config_file="", seed=0):
        self.seed = seed

    def generate_qa_pairs(self, prompts, model_list):
        pipeline_dict = {}
        for model in model_list:
            qa_pairs = feed_forward(model, prompts)
            pipeline_dict[model] = [{"qa_pair": qa_pair} for qa_pair in qa_pairs]

        return pipeline_dict
    
    def judge_prompts(prompt_type="", category="",  pipeline_dict=None, judge_template="", judge_model="gpt-4-turbo"):
        if prompt_type == "harmful":
            prompt_template, _ = load_prompt_format(judge_template, 'base-#thescore')
        elif prompt_type == "over_cautious":
            prompt_template, _ = load_prompt_format(judge_template, 'base-general-v3')
            
        templated_qa_pairs = []
        for model in pipeline_dict.keys():
            for object in pipeline_dict[model]:
                templated_qa = prompt_template %  object["qa_pair"]
                templated_qa_pairs.append(templated_qa)
            
            from gpt_batch.batcher import GPTBatcher

            # Initialize the batcher
            batcher = GPTBatcher(api_key=os.getenv("OPENAI_API_KEY"), model_name='gpt-4-turbo')

            # Send a list of messages and receive answers
            judge_results = batcher.handle_message_list(templated_qa_pairs)
            
            pipeline_dict[model] = [{"qa_pair": qa_pair, "judge_result": judge_result} for qa_pair, judge_result in zip(judge_results,pipeline_dict[model])]
        
        return pipeline_dict
    
    def rank_rejected_prompts(self, top_k=10, rejected_dict=None, ):
        
    def create_woke_data(input_dict, generation_model="gpt-4-turbo"):