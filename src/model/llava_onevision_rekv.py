import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.vidcom2_for_rekv import *

# from model.custom_siglip import custom_SiglipEncoder
import time
# from model.experiment.after_vit_compression.dpc_knn import *
# from model.experiment.sttm import *
# 
# from model.data_store import data
# from model.visualization import *
import os
class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        
        
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        
        
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]
        frames=selected_video_feature.shape[0]
        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        ####################################################################
        
        # reshaped_video_tensor=video_features.reshape(-1, video_features.size(-1))  
        # droped_video_features=vidcom2_compression(self,reshaped_video_tensor) #In [1]: video_features.shape  Out[1]: torch.Size([64, 196, 3584])-》 torch.Size([6272, 3584])
        
        
        
        # droped_video_features=dynamic_processor(self,reshaped_video_tensor)

        
        # droped_video_features=reduce_tokens_per_frame_dbdpc(video_features)
        # droped_video_features, masks, weights = frame_wise_token_reduction(
        # video_features, 
        # reduction_ratio=0.5,  # 保留50%的tokens
        # dc=0.2,  # 可以根据需要调整
        # print_debug=True
        # )
        
        # droped_video_features=spatial_token_merging_with_budget(video_features, 98)

        # droped_video_features=dpc_knn_token_reducer_strict(video_features) #In [1]: video_features.shape  Out[1]: torch.Size([64, 196, 3584])-》 torch.Size([6272, 3584])
        
        # ####################################################################
        token_per_frame = 196
        
        # kept_indices_tensor=data.kept_indices_tensor
        # start_indices=data.start_indices
        
        
        # denormalized_video_for_viz = de_normalize_video_rgb(pixel_values_videos)
        # from IPython import embed;embed()
        
        # visualize_token_pruning(original_video=denormalized_video_for_viz,indices_to_keep_per_frame=kept_indices_tensor,base_frame_indices=start_indices,patch_size=14)

        

        video_features = video_features.reshape(batch_size, frames * token_per_frame, -1)  # (B, Nv*196, D)
        
        # video_features = video_features.reshape(batch_size, frames * token_per_frame, -1)  # (B, Nv*196, D)
        
        
        ###################################################################################################################
        return video_features

    # @torch.inference_mode()
    # def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
    #     device = self.device
    #     stop_token_ids = [self.processor.tokenizer.eos_token_id]

    #     output_ids = []
    #     stopped = False

    #     # NOTE: Only input the question to perform retrieval.
        
        
    #     # NOTE: Only input the question to perform retrieval.
    #     # Handle both string input and dict input
    #     if isinstance(input_text, str):
    #         question_text = input_text
    #         prompt_text = input_text
    #     else:
    #         # Original dict-based approach
    #         question_text = input_text['question']
    #         prompt_text = input_text['prompt']
            
    #     input_ids = self.processor.tokenizer(question_text).input_ids
        
        
    #     input_ids = torch.as_tensor([input_ids], device=device)
    #     for layer_kv in self.kv_cache:  # activate retrieval mode
    #         layer_kv.set_retrieval()

    #     if retrieved_indices is None:  # Internal retrieval
    #         out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
    #         past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
    #     else:  # External retrieval
    #         for layer_kv in self.kv_cache:
    #             assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
    #             layer_kv.set_retrieved_block_indices(retrieved_indices)
    #         out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
    #         past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

    #     for layer_kv in self.kv_cache:  # reset to default
    #         layer_kv.reset_retrieval()

    #     for i in range(max_new_tokens):
    #         if i == 0:  # prefill
    #             input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
    #             input_ids = torch.as_tensor([input_ids], device=device)
    #             inputs_embeds = self.get_input_embeddings()(input_ids)
    #             out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
    #             past_key_values = out.past_key_values
    #             logits = out.logits
    #         else:  # decoding
    #             out = self.language_model(
    #                 input_ids=torch.as_tensor(
    #                     [[token]],
    #                     device=device,
    #                 ),
    #                 use_cache=True,
    #                 past_key_values=past_key_values,
    #             )
    #             logits = out.logits
    #             past_key_values = out.past_key_values

    #         last_token_logits = logits[0, -1, :]
            
    #         _, indices = torch.topk(last_token_logits, 2)
    #         tokens = [int(index) for index in indices.tolist()]
    #         token = tokens[0]

    #         output_ids.append(token)

    #         if token in stop_token_ids:
    #             stopped = True
    #         else:
    #             stopped = False

    #         if i == max_new_tokens - 1 or stopped:
    #             break

    #     output = self.processor.tokenizer.decode(
    #         output_ids,
    #         skip_special_tokens=True,
    #         spaces_between_special_tokens=False,
    #         clean_up_tokenization_spaces=True,
    #     )
        
    #     return output
    
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        # Handle both string input and dict input
        if isinstance(input_text, str):
            question_text = input_text
            prompt_text = input_text
        else:
            # Original dict-based approach
            question_text = input_text['question']
            prompt_text = input_text['prompt']
            
        input_ids = self.processor.tokenizer(question_text).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        
        for layer_kv in self.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(prompt_text).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break
        
        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output

def load_model(model_path='/data/u_2359629761/yy_datasets/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068',
accelerator=None, local_rank=None, n_init=None, n_local=15000, topk=64, chunk_size=1):
    """
    load_model 支持传入 accelerate.Accelerator 或 local_rank（int）。
    - 如果传入 accelerator，则优先使用 accelerator.local_process_index 将模型加载到对应 GPU（device_map 指定为 'cuda:{local_rank}'）
    - 如果未传入 accelerator，但提供了 local_rank，则使用 'cuda:{local_rank}'
    - 否则退回到单 GPU（cuda:0）行为
    注意：我们不再使用 device_map="auto"，以避免 HF 自动切分到所有可见显卡。
    """
    # 1) 决定设备字符串
    if accelerator is not None:
        local_rank = accelerator.local_process_index
        device = torch.device(f"cuda:{local_rank}")
    elif local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")


    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
    'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
    'n_local': n_local,
    'fattn': True,
    'block_size': n_frame_tokens,
    'topk': topk,
    'chunk_size': chunk_size,
    'max_cached_block': 128,
    'exc_block_size': n_frame_tokens,
    'pin_memory': True,
    }


    # 2) 从权重加载模型：禁用 device_map='auto'，并指定整个模型放到当前进程的 GPU（避免全局分布式切分）
    model = LlavaOneVision_ReKV.from_pretrained(
    model_path,
    device_map={"": f"cuda:{device.index}"},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    processor=processor,
    n_frame_tokens=n_frame_tokens,
    init_prompt_ids=init_prompt_ids,
    n_local=n_local,
    topk=topk,
    chunk_size=chunk_size,
    )


    # 确认模型在目标 device 上
    try:
        model.to(device)
    except Exception:
        # 有些 HF API 已经把模型放到 device_map 指定的显卡，这里忽略异常
        pass


    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')


    model.eval()


    return model, processor