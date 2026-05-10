import os
import time

# 1. 优先设置国内镜像 (必须在 import 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

# 2. 配置参数
repo_id = "FoundationVision/InfinityStar"
local_dir = "/home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/checkpoint"

print(f"🚀 开始下载 InfinityStar 模型权重...")
print(f"📂 目标目录: {local_dir}")
print(f"🔗 镜像源: {os.environ['HF_ENDPOINT']}")

# 3. 自动重试循环
while True:
    try:
        print("\n🔄 正在扫描并下载 (支持断点续传)...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 下载真实文件
            resume_download=True,         # 确保断点续传
            max_workers=4                 # 开启并发下载
        )
        
        print("\n🎉🎉🎉 恭喜！InfinityStar 模型下载全部完成！")
        break # 成功退出

    except Exception as e:
        # 简单的错误捕获，防止特殊字符报错
        err_msg = str(e).replace("'", "").replace('"', "")[:100]
        print(f"\n⚠️ 网络中断或报错: {err_msg}")
        print("🔄 10秒后自动重试，请勿关闭窗口...")
        time.sleep(10)
