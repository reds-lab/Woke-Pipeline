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
    
    def judge_prompts(prompt_type="", category="",  pipeline_dict=None, prompt_template="", judge_model="gpt-4-turbo"):
        if pipeline_dict == None:
            raise ValueError("pipeline_dict needs to be inputted")
        
        if prompt_type == "harmful":
            prompt_template, _ = load_prompt_format(name='base-#thescore')
        elif prompt_type == "over_cautious":
            prompt_template, _ = load_prompt_format(name='base-general-v3')

        if prompt_template != "":
            prompt_template = prompt_template
        
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
            
            #Merge the dict
            pipeline_dict[model] = [dict(object, judge_result=judge_result) for object, judge_result in zip(pipeline_dict[model], judge_results)]
        
        return pipeline_dict
    
    def rank_rejected_prompts(self, top_k=10, pipeline_dict=None):
        """
            Incredibly time consuming. Optimization is a work in progress.
        """
        

    def create_woke_data(input_dict, generation_model="gpt-4-turbo"):
