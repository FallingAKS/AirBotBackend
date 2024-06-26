# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Modified on: https://github.com/yangjianxin1/GPT2-chitchat
"""
import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel

PAD = '[PAD]'
pad_id = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
# 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = '8bd4e1c5'
SPARKAI_API_SECRET = 'ZDc4NDA0ODhjY2FhMmUyNWE0YTg1ZGNj'
SPARKAI_API_KEY = '582e5780be2b0948ba1af98d2c3b9f21'
# 星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'


class Inference:
    def __init__(
            self,
            model_name_or_path,
            device=device,
            max_history_len=10,
            max_len=100,
            repetition_penalty=1.0,
            temperature=1.0,
            topk=8,
            topp=0.0
    ):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        # 存储聊天记录，每个utterance以token的id的形式进行存储
        self.history = []
        self.max_history_len = max_history_len
        self.max_len = max_len
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

    def predict(self, query, use_history=True):
        text_ids = self.tokenizer.encode(query, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        if use_history:
            for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(self.tokenizer.sep_token_id)
        else:
            input_ids.extend(text_ids)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long().to(self.device)
        input_ids = input_ids.unsqueeze(0)
        response = []  # 根据context，生成的response
        for _ in range(self.max_len):  # 最多生成max_len个token
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(response):
                next_token_logits[id] /= self.repetition_penalty
            next_token_logits = next_token_logits / self.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        self.history.append(response)
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return "".join(response_tokens)

    def restart(self):
        self.history = []


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, help='最高积累概率')
    parser.add_argument('--log_path', default='interact.log', type=str, help='interact日志存放位置')
    parser.add_argument('--model_dir', default='./outputs/min_ppl_model/', type=str, help='对话模型文件夹路径')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--max_len', type=int, default=100, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=10, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


history = []
title = ""


def interact(input: str, model_dir, max_history_len, max_len, repetition_penalty, temperature):
    global history
    global title
    inference = Inference(model_dir, device, max_history_len, max_len, repetition_penalty,
                          temperature)

    if input is None or input == "":
        return "请输入问题！"
    elif input == "raks":
        # 重启对话
        inference.restart()
        history = []
        title = ""
        return "aks: restart successfully!"

    while True:
        try:
            query = "乘客:" + input
            input_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            text = inference.predict(query)

            if title is None or title == "":
                spark = ChatSparkLLM(
                    spark_api_url=SPARKAI_URL,
                    spark_app_id=SPARKAI_APP_ID,
                    spark_api_key=SPARKAI_API_KEY,
                    spark_api_secret=SPARKAI_API_SECRET,
                    spark_llm_domain=SPARKAI_DOMAIN,
                    streaming=False,
                )
                messages = [ChatMessage(
                    role="user",
                    content='根据user【' + input + '】的提问和assistant【' + text + '】的回答为对话起一个标题，返回答案时只需返回对话标题即可。',
                )]
                handler = ChunkPrintHandler()
                input_string = str(spark.generate([messages], callbacks=[handler]))

                # 使用 split 方法提取 text 内容
                start = input_string.find("text='") + len("text='")
                end = input_string.find("'", start)
                title = input_string[start:end]

            history.append({
                'choices': [
                    {
                        'message': {
                            'content': input,
                            'role': 'user'
                        }
                    }
                ],
                'createdTime': input_time
            })
            text_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            response = {
                'choices': [
                    {
                        'message': {
                            'content': text,
                            'role': 'assistant'
                        }
                    }
                ],
                'createdTime': text_time
            }
            history.append(response)

            return response
        except ValueError:
            break


def get_history():
    global history
    if history is None or len(history) == 0:
        return []
    return history


def get_title():
    global title
    if title is None or title == "":
        return "No title"
    return title
