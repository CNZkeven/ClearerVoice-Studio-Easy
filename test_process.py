# -*- coding: utf-8 -*-
"""测试音频处理功能"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'clearvoice'))

from clearvoice.network_wrapper import network_wrapper

print("开始测试音频处理...")

# 加载模型
wrapper = network_wrapper()
model = wrapper('speech_enhancement', 'MossFormer2_SE_48K')

if model is None:
    print("错误：模型加载失败")
    sys.exit(1)

print(f'模型加载成功：{type(model)}')
print(f'有 process 方法：{hasattr(model, "process")}')
print(f'有 write 方法：{hasattr(model, "write")}')

# 检查 process 方法是否可以调用
import inspect
process_sig = inspect.signature(model.process)
print(f'process 方法签名：{process_sig}')

print("测试完成!")
