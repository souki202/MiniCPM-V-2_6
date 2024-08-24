import gradio as gr
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import threading
import os
import asyncio
import argparse

model_path = './'

# モデルとトークナイザーの読み込み関数
def load_model():
    global model, tokenizer, model_loaded
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_loaded = True

# モデルとトークナイザーの読み込みをバックグラウンドで実行
model_loaded = False
threading.Thread(target=load_model).start()

async def predict(image, question):
    while not model_loaded:
        await asyncio.sleep(1)
    
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    return res

async def process_folder(folder_path, question):
    if not model_loaded:
        return "Model is still loading, please wait..."
    
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            result = await predict(image, question)
            output_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            results.append(f"{filename}: {result}")
    return "\n".join(results)

async def process_single_image(image, question):
    if not model_loaded:
        return "Model is still loading, please wait..."
    result = await predict(image, question)
    return result

# Gradioインターフェースの設定
with gr.Blocks() as demo:
    with gr.Tab("Batch Processing"):
        gr.Markdown("## Batch Image Question Answering with MiniCPM-V 2.6")
        folder_path = gr.Textbox(lines=1, placeholder="Enter folder path here...", label="Folder Path")
        question_batch = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
        batch_output = gr.Textbox(label="Output")
        batch_button = gr.Button("Process Folder")
        batch_button.click(fn=process_folder, inputs=[folder_path, question_batch], outputs=batch_output)
    
    with gr.Tab("Single Image Processing"):
        gr.Markdown("## Single Image Captioning with MiniCPM-V 2.6")
        image = gr.Image(type="pil", label="Upload Image")
        question_single = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
        single_output = gr.Textbox(label="Output")
        single_button = gr.Button("Process Image")
        single_button.click(fn=process_single_image, inputs=[image, question_single], outputs=single_output)

# コマンドライン引数の解析
parser = argparse.ArgumentParser(description="Launch Gradio interface with specified server name and port.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server name to use for Gradio interface.")
parser.add_argument("--port", type=int, default=7860, help="Server port to use for Gradio interface.")
args = parser.parse_args()

# インターフェースの起動
demo.launch(server_name=args.host, server_port=args.port)