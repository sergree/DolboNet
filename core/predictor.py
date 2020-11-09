# Модуль загрузки и семплирования из Transformer
# by Sergree
# https://github.com/sergree

import numpy as np
from scipy.special import softmax
from core.tf_transformer import transformer
import config
from core.yadisk import download_yadisk_link
from utils.tprint import log
from tqdm import tqdm

# Параментры Transformer взяты из оригинальной публикации:
# https://arxiv.org/abs/1706.03762 (стр. 9 - base)

NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.1

log(f"Загружаю {config.weights_file}...")

model = transformer(
    vocab_size=config.vocab_size,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
)
try:
    model.load_weights(config.weights_file)
except OSError:
    log(f"Похоже весов нет! Попробую скачать с Яндекс.Диска, подождите 2 минуты...")
    with open("weights/weights.txt") as f:
        url = f.readline().strip()
        download_yadisk_link(url, filename=config.weights_file)
        model.load_weights(config.weights_file)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

log(f"{config.weights_file} загружен.")


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = preds / temperature
    preds = softmax(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def decode_sequence(input_seq, temperature, prewarm=False):
    target_seq = np.zeros((1, 1), dtype="uint16")
    target_seq[0, 0] = 2
    stop_condition = False
    decoded_sentence = []
    if prewarm:
        pbar = tqdm(total=config.max_len)
    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])
        sampled_token_index = sample(output_tokens[0, -1, :], temperature=temperature)
        decoded_sentence.append(sampled_token_index)
        if len(decoded_sentence) > config.max_len:
            stop_condition = True
        elif sampled_token_index == 4 and not prewarm:
            stop_condition = True
        packed_sampled_token_index = np.zeros((1, 1))
        packed_sampled_token_index[0, 0] = sampled_token_index if not prewarm else 1
        target_seq = np.append(target_seq, packed_sampled_token_index, axis=-1)
        if prewarm:
            pbar.update(1)
            if stop_condition:
                pbar.close()
    return decoded_sentence


if config.use_prewarm:
    log(
        "Подготовительный прогон трансформера пустыми данными (читайте why_prewarm.txt)..."
    )
    decode_sequence(
        np.ones((1, config.max_len), dtype="uint16"), config.temperature, prewarm=True
    )
    log("Прогон трансформера завершен.")
