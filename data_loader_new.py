# -*- coding: utf-8 -*-
# Import thư viện
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def get_text_pairs(file_path, limit=None):
    with open(file_path, encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]
    text_pairs = []
    if limit is None:
        limit = len(lines)
    i = 0
    for line in lines:
        stripped, original = line.split('\t')
        original = '[start] ' + original
        text_pairs.append((stripped, original))
        i += 1
        if i >= limit:
            break
    return text_pairs


def split_pairs(text_pairs, ratio=.15, shuffle=False):
    if shuffle:
        random.shuffle(text_pairs)

    num_val_samples = int(ratio * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples+num_val_samples]
    test_pairs = text_pairs[num_train_samples+num_val_samples:]
    return train_pairs, val_pairs, test_pairs


def load_data(file_path, limit=None, ratio=.15, shuffle=False):
    text_pairs = get_text_pairs(file_path, limit)
    return split_pairs(text_pairs, ratio, shuffle)


def create_vectorizations(train_pairs, sequence_length=50, vocab_size=15000):
    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    train_stripped_texts = [n[0] for n in train_pairs]
    source_vectorization.adapt(train_stripped_texts)
    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length+1
    )
    train_original_texts = [n[1] for n in train_pairs]
    target_vectorization.adapt(train_original_texts)
    return source_vectorization, target_vectorization


def save_vectorization(vectorization, file_path):
    '''
    Save the config of a vectorization to disk as a pickle file.
    '''
    with open(file_path, 'wb') as f:
        f.write(pickle.dumps({'config': vectorization.get_config()}))
def load_vectorization_from_disk(vectorization_path):
    '''
    Load a saved vectorization from disk.
    '''
    with open(vectorization_path, 'rb') as f:
        from_disk = pickle.load(f)
        new_v = layers.TextVectorization(
            max_tokens=from_disk['config']['max_tokens'],
            output_mode='int',
            output_sequence_length=from_disk['config']['output_sequence_length']
        )
        # Adapt with dummy data to initialize the layer properly
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["dummy data"]))
    return new_v
def format_dataset(stripped, source_vectorization, original, target_vectorization):
    strip = source_vectorization(stripped)
    origin = target_vectorization(original)

    return ({
        'stripped': strip,
        'original': origin[:, :-1]
    }, origin[:, 1:])


def make_dataset(pairs, source_vectorization, target_vectorization, batch_size):
    stripped_texts, original_texts = zip(*pairs)
    stripped_texts = list(stripped_texts)
    original_texts = list(original_texts)
    dataset = tf.data.Dataset.from_tensor_slices((stripped_texts, original_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: format_dataset(x, source_vectorization, y, target_vectorization), num_parallel_calls=8)
    return dataset.shuffle(2048).prefetch(16).cache()
