# -*- coding: utf-8 -*-
import sys
import os

# 设置路径
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'clearvoice'))

from clearvoice.network_wrapper import network_wrapper

print("开始加载模型...")
wrapper = network_wrapper()
model = wrapper('speech_enhancement', 'MossFormer2_SE_48K')

if model is None:
    print("错误：模型为 None")
    sys.exit(1)

print('Model loaded:', model)
print('Model.model:', model.model)
print('checkpoint_dir:', model.args.checkpoint_dir)

# 检查模型文件路径是否存在
checkpoint_file = os.path.join(model.args.checkpoint_dir, 'last_best_checkpoint')
print(f'checkpoint_file exists: {os.path.exists(checkpoint_file)}')

# 检查模型权重
model_file_path = os.path.join(model.args.checkpoint_dir, 'last_best_checkpoint')
if os.path.exists(model_file_path):
    with open(model_file_path, 'r') as f:
        model_file_name = f.readline().strip()
    full_path = os.path.join(model.args.checkpoint_dir, model_file_name)
    print(f'模型权重文件：{full_path}')
    print(f'模型权重文件存在：{os.path.exists(full_path)}')

print("测试完成!")
