# package_name/my_class.py
import os

from feedforward import feed_forward


class WokePipeline:
    def __init__(self, API_KEY="", seed=0):
        self.API_KEY = seed
        self.seed = seed

    def generate(self, harmful_seeds, model_list):
        response_dict = {}
        for model in model_list:
            prompts = feed_forward(model, harmful_seeds)
            response_dict[model] = prompts

        for model in response_dict.keys():
            prompts
        
        QA_Pair = (raw_prompt, answer)
        model = "gpt-4-turbo"
        prompt_template, outputformat = load_prompt_format(judge_prompt_filename, 'base-#thescore')

        content = prompt_template % QA_Pair
    
        from gpt_batch.batcher import GPTBatcher

        # Initialize the batcher
        batcher = GPTBatcher(api_key=os.getenv("OPENAI_API_KEY"), model_name='gpt-4-turbo')

        # Send a list of messages and receive answers
        result = batcher.handle_message_list(['question_1', 'question_2', 'question_3', 'question_4'])
        print(result)
        # Expected output: ["answer_1", "answer_2", "answer_3", "answer_4"]