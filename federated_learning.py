import copy
import torch
import torch.nn.functional as F
from typing import List

def add_noise(model: torch.nn.Module, noise_scale: float) -> torch.nn.Module:
    """
    对模型参数施加差分隐私噪声。
    """
    noisy_model = copy.deepcopy(model)
    for param in noisy_model.parameters():
        param.data += torch.randn_like(param) * noise_scale
    return noisy_model

def federated_distillation_with_dp(client_models: List[torch.nn.Module], temperature: float, noise_scale: float) -> torch.nn.Module:
    """
    使用联邦蒸馏、差分隐私和FedAvg聚合多个客户端模型,得到全局模型。
    """
    global_model = None

    # 计算客户端模型的softmax输出
    client_outputs = []
    for client_model in client_models:
        # 对每个客户端模型添加差分隐私噪声
        noisy_client_model = add_noise(client_model, noise_scale)
        client_output = F.softmax(noisy_client_model(input_batch) / temperature, dim=1)
        client_outputs.append(client_output)

    # 计算全局模型的softmax输出
    global_output = torch.stack(client_outputs, dim=0).mean(dim=0)

    # 使用FedAvg聚合客户端模型,得到全局模型
    global_model = copy.deepcopy(client_models[0])
    for param in global_model.parameters():
        param.data = torch.stack([client_model.state_dict()[param.name] for client_model in client_models], dim=0).mean(dim=0)

    # 训练全局模型
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    for epoch in range(10):
        optimizer.zero_grad()
        output = F.softmax(global_model(input_batch) / temperature, dim=1)
        loss = F.kl_div(output, global_output, reduction='batchmean')
        loss.backward()
        optimizer.step()

    return global_model