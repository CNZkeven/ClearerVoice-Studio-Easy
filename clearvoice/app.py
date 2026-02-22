# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
from flask import Flask, render_template, request, jsonify, send_file, after_this_request
from flask_cors import CORS
import threading
import uuid
import shutil
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from clearvoice import ClearVoice
from clearvoice.network_wrapper import network_wrapper
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

PROJECT_ROOT = parent_dir
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
TEMP_DIR = os.path.join(current_dir, 'temp')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_task = None
        self.model_config = {}
        self.is_loaded = False

    def set_config(self, task, model_name, config):
        self.current_task = task
        self.model_name = model_name
        self.model_config = config
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return True

        try:
            wrapper = network_wrapper()
            self.current_model = wrapper(self.current_task, self.model_name)

            if self.current_model and self.current_model.model:
                self.is_loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_audio(self, input_path, output_path):
        if not self.is_loaded:
            if not self.load_model():
                return False, "Model load failed"

        try:
            # Use process method instead of calling model directly
            output_wav = self.current_model.process(input_path, online_write=False)
            self.current_model.write_audio(output_path, audio=output_wav)
            return True, output_path
        except Exception as e:
            return False, str(e)

model_manager = ModelManager()

TASK_MODELS = {
    "speech_enhancement": [
        {"name": "MossFormer2_SE_48K", "desc": "MossFormer2 SE (48k)", "sampling_rate": 48000},
        {"name": "FRCRN_SE_16K", "desc": "FRCRN SE (16k)", "sampling_rate": 16000},
        {"name": "MossFormerGAN_SE_16K", "desc": "MossFormerGAN SE (16k)", "sampling_rate": 16000}
    ],
    "speech_separation": [
        {"name": "MossFormer2_SS_16K", "desc": "MossFormer2 SS (16k)", "sampling_rate": 16000, "num_spks": 2}
    ],
    "speech_super_resolution": [
        {"name": "MossFormer2_SR_48K", "desc": "MossFormer2 SR (8k->48k)", "sampling_rate": 48000}
    ],
    "target_speaker_extraction": [
        {"name": "AV_MossFormer2_TSE_16K", "desc": "AV-MossFormer2 TSE (16k)", "sampling_rate": 16000}
    ]
}

DEFAULT_CONFIGS = {
    "MossFormer2_SE_48K": {
        "one_time_decode_length": 20,
        "decode_window": 4,
        "win_type": "hamming",
        "win_len": 1920,
        "win_inc": 384,
        "fft_len": 1920,
        "num_mels": 60
    },
    "FRCRN_SE_16K": {
        "one_time_decode_length": 120,
        "decode_window": 1,
        "win_type": "hanning",
        "win_len": 640,
        "win_inc": 320,
        "fft_len": 640
    },
    "MossFormerGAN_SE_16K": {
        "one_time_decode_length": 10,
        "decode_window": 10,
        "win_type": "hamming",
        "win_len": 400,
        "win_inc": 100,
        "fft_len": 400
    },
    "MossFormer2_SS_16K": {
        "one_time_decode_length": 2,
        "decode_window": 2,
        "num_spks": 2,
        "encoder_kernel_size": 16,
        "encoder_embedding_dim": 512,
        "mossformer_sequence_dim": 512,
        "num_mossformer_layer": 24
    },
    "MossFormer2_SR_48K": {
        "one_time_decode_length": 20,
        "decode_window": 4
    },
    "AV_MossFormer2_TSE_16K": {
        "one_time_decode_length": 3,
        "decode_window": 3,
        "network_reference_cue": "lip",
        "network_reference_backbone": "resnet18",
        "network_reference_emb_size": 256,
        "network_audio_backbone": "mossformer2",
        "network_audio_encoder_kernel_size": 16,
        "network_audio_encoder_out_nchannels": 512,
        "network_audio_encoder_in_nchannels": 1,
        "network_audio_masknet_numspks": 1,
        "network_audio_masknet_chunksize": 250,
        "network_audio_masknet_numlayers": 1,
        "network_audio_masknet_norm": "ln",
        "network_audio_intra_numlayers": 24,
        "network_audio_intra_nhead": 8,
        "network_audio_intra_dffn": 1024,
        "network_audio_intra_dropout": 0
    }
}

def get_model_list():
    return TASK_MODELS

def get_default_config(model_name):
    return DEFAULT_CONFIGS.get(model_name, {})

def get_checkpoints_dir():
    return CHECKPOINT_DIR

def set_checkpoints_dir(new_dir):
    global CHECKPOINT_DIR
    if os.path.isdir(new_dir):
        CHECKPOINT_DIR = new_dir
        return True
    return False

def get_installed_models():
    installed = []
    if os.path.exists(CHECKPOINT_DIR):
        for model_name in DEFAULT_CONFIGS.keys():
            model_path = os.path.join(CHECKPOINT_DIR, model_name)
            if os.path.isdir(model_path):
                if os.listdir(model_path):
                    installed.append(model_name)
    return installed

def download_model_from_huggingface(model_name):
    try:
        from huggingface_hub import snapshot_download
        model_path = os.path.join(CHECKPOINT_DIR, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print(f"Downloading model {model_name} from HuggingFace...")
        snapshot_download(repo_id=f"alibabasglab/{model_name}", local_dir=model_path)
        return True, "Download successful"
    except Exception as e:
        return False, str(e)

@app.route('/')
def index():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/tasks')
def get_tasks():
    return jsonify(list(TASK_MODELS.keys()))

@app.route('/api/models/<task>')
def get_models(task):
    if task in TASK_MODELS:
        return jsonify(TASK_MODELS[task])
    return jsonify([])

@app.route('/api/config/<model_name>')
def get_config(model_name):
    config = get_default_config(model_name)
    return jsonify(config)

@app.route('/api/checkpoint_dir', methods=['GET', 'POST'])
def checkpoint_dir():
    if request.method == 'GET':
        return jsonify({"checkpoint_dir": CHECKPOINT_DIR})
    else:
        data = request.json
        new_dir = data.get('path', '')
        if set_checkpoints_dir(new_dir):
            return jsonify({"success": True, "checkpoint_dir": CHECKPOINT_DIR})
        return jsonify({"success": False, "error": "Invalid directory"})

@app.route('/api/installed_models')
def get_installed():
    return jsonify(get_installed_models())

@app.route('/api/model_info')
def model_info():
    installed = get_installed_models()
    return jsonify({
        "installed": installed,
        "available": list(DEFAULT_CONFIGS.keys()),
        "checkpoint_dir": CHECKPOINT_DIR
    })

@app.route('/api/load_model', methods=['POST'])
def load_model():
    data = request.json
    task = data.get('task')
    model_name = data.get('model_name')
    config = data.get('config', {})

    model_manager.set_config(task, model_name, config)

    if model_manager.load_model():
        return jsonify({"success": True, "message": f"Model {model_name} loaded successfully"})
    else:
        return jsonify({"success": False, "error": "Model load failed, please ensure model is downloaded"})

@app.route('/api/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Please upload audio file"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})

    task = request.form.get('task', 'speech_enhancement')
    model_name = request.form.get('model_name', 'MossFormer2_SE_48K')
    config_str = request.form.get('config', '{}')

    try:
        config = json.loads(config_str)
    except:
        config = {}

    model_manager.set_config(task, model_name, config)

    temp_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{temp_id}_input_{file.filename}")
    output_path = os.path.join(TEMP_DIR, f"{temp_id}_output_{file.filename}")

    file.save(input_path)

    success, result = model_manager.process_audio(input_path, output_path)

    try:
        os.remove(input_path)
    except:
        pass

    if success:
        @after_this_request
        def remove_output_file(response):
            try:
                threading.Timer(300, lambda: os.remove(output_path) if os.path.exists(output_path) else None).start()
            except:
                pass
            return response
        return jsonify({
            "success": True,
            "output_file": f"/api/download/{os.path.basename(output_path)}",
            "temp_path": output_path
        })
    else:
        return jsonify({"success": False, "error": result})

@app.route('/api/download/<filename>')
def download_file(filename):
    filepath = os.path.join(TEMP_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

@app.route('/api/download_model', methods=['POST'])
def download_model():
    data = request.json
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({"success": False, "error": "Model name is required"})

    if model_name not in DEFAULT_CONFIGS:
        return jsonify({"success": False, "error": "Unknown model"})

    success, message = download_model_from_huggingface(model_name)

    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "error": message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
