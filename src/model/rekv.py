# ===== file: model/rekv.py =====
from operator import attrgetter


import torch
import os
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
from model.video_qa.rekv_offline_vqa import ReKVOfflineVQA


from model.llava_onevision_rekv import load_model
warnings.filterwarnings("ignore")




from model.modelclass import Model
class rekv(ReKVOfflineVQA):
    def __init__(self, accelerator=None):
        # 不在模块导入阶段做全量模型加载，支持传入 accelerate 的 accelerator
        self._accelerator = accelerator
        # 延迟加载：如果在构造时提供了 accelerator，就立即加载模型
        if accelerator is not None:
            ReKV_Init(self)


    def Run(self, file, inp):
        return ReKV_Run(self, file, inp)
    def name(self):
        return "rekv"


# 提供手动初始化接口，供 main.py 在创建 accelerator 后调用
    def initialize(self, accelerator=None, model_path=None):
        if accelerator is not None:
            self._accelerator = accelerator
        if model_path is not None:
            self._model_path = model_path
        ReKV_Init(self) 
def ReKV_Init(self):
    # Load the OneVision model
    self.qa_model, self.processor = load_model(accelerator=self._accelerator)

def ReKV_Run(self,file, inp):
    self.qa_model.past_memory_mean_token=[]
    video = self.load_video(file)
    if not isinstance(video, torch.Tensor):
        video_tensor = torch.from_numpy(video)
    else:
        video_tensor = video
    self.qa_model.clear_cache()
    self.qa_model.encode_init_prompt()
    self.qa_model.encode_video(video_tensor)
    response=self.qa_model.question_answering(inp)
    print("response:",response)
    return response

