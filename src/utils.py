import os

import torch


def setup_environment():
    """環境変数の設定を行います"""
    # 最もメモリが空いているGPUを使用
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # 各GPUの空きメモリを確認
            free_memory = []
            for i in range(device_count):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory.append(total_memory - allocated_memory)

            # 最も空きメモリが多いGPUのインデックスを取得
            best_gpu = free_memory.index(max(free_memory))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"最もメモリが空いているGPU {best_gpu} を使用します")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("GPUが利用できません")


def print_gpu_info():
    """GPUの情報を表示します"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )


def print_gpu_memory():
    """GPUのメモリ使用量を表示します"""
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

