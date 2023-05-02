import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm

input_file_path = os.path.join(os.path.dirname(__file__), 'dataset/DATASET.txt')

if not os.path.exists(input_file_path):
    raise FileNotFoundError("La carpeta no se encuentra en la ruta especificada")

enc = tiktoken.get_encoding("gpt2")
chunk_size = 100 * 1024 * 1024  # 100 MB

train_file_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.bin')

file_size = os.path.getsize(input_file_path)
progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing")

with open(train_file_path, 'wb') as train_file, open(val_file_path, 'wb') as val_file:
    with open(input_file_path, 'r') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            n = len(data)
            train_data = data[:int(n*0.9)]
            val_data = data[int(n*0.9):]
            train_tokens = enc.encode_ordinary(train_data)
            val_tokens = enc.encode_ordinary(val_data)

            train_tokens = np.array(train_tokens, dtype=np.uint16)
            val_tokens = np.array(val_tokens, dtype=np.uint16)

            train_tokens.tofile(train_file)
            val_tokens.tofile(val_file)
            
            progress_bar.update(len(data))

progress_bar.close()
