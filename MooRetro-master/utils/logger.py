import logging
import os
import time
from datetime import datetime

# 配置日志系统
def setup_logger(log_dir="logs"):
    """配置日志系统"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"andor_tree_{timestamp}.log")
    
    logger = logging.getLogger("ANDORTree")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s] [%(filename)-20s:%(lineno)3d] [%(levelname)-8s] %(message)s')
        # '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] [%(filename)-20s:%(lineno)3d] [%(levelname)-8s] %(message)s')
        # '%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger