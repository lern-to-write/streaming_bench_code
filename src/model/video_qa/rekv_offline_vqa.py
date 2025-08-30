import torch
from logzero import logger

from model.video_qa.base import BaseVQA, work

from model.video_qa.run_eval import time_recoder

class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        
        

        
        self.qa_model.past_memory_mean_token=[]
        video = self.load_video(video_path)
        

        
        
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        ########################################################
        torch.cuda.reset_peak_memory_stats()
        gen_start_event = torch.cuda.Event(enable_timing=True)
        gen_end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        gen_start_event.record()
        ######################################################
        
        self.qa_model.encode_video(video_tensor)
        
        # #############################################################
        gen_end_event.record()
        torch.cuda.synchronize()
        
        gen_time = gen_start_event.elapsed_time(gen_end_event) / 1000.0  # 秒
        gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        time_recoder.total_cuda_time += gen_time
        time_recoder.max_mem=max(gen_max_mem,time_recoder.max_mem)
        print("total_time",time_recoder.total_cuda_time,"max_mem",time_recoder.max_mem)
        ###########################################################
        


        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            
            # QA
            if 'choices' in sample:  # CloseQA
                choices = sample['choices']
                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                })
            else:  # OpenQA
                qa_results = self.video_open_qa(question)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']
                
                



if __name__ == "__main__":
    work(ReKVOfflineVQA)
