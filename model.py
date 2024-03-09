import json
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load data
def load_data(data_path):
    with open(os.path.join(data_path, 'dataset_rsicd.json'), 'r') as f:
        data = json.load(f)
    return data['images']

def load_and_preprocess_images(image_paths, target_size):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image = np.array(image)
        image = preprocess_input(image)
        images.append(image)
    return np.array(images)

def preprocess_data(images_data):
    captions = []
    images = []
    for image_data in images_data:
        for sentence in image_data['sentences']:
            captions.append(sentence['raw'])
            images.append(image_data['filename'])
    return images, captions

data_path = "annotations_rsicd"
image_directory = "RSICD_images"
images_data = load_data(data_path)
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory)]

# Preprocess data
images, captions = preprocess_data(images_data)

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
caption_sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(seq) for seq in caption_sequences)
padded_sequences = pad_sequences(caption_sequences, maxlen=max_length, padding='post')

# Define model
def define_model(image_shape, vocab_size, max_length):
    image_input = Input(shape=image_shape)
    encoder = ResNet50(weights='imagenet', include_top=False, input_shape=image_shape)
    encoder.trainable = False
    encoded_image = encoder(image_input)
    
    decoder_input = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, 256, input_length=max_length, mask_zero=True)(decoder_input)
    decoder_lstm = LSTM(256, return_sequences=True)(embedding_layer)
    attention = Attention()([encoded_image, decoder_lstm])
    decoder_concat = Concatenate()([decoder_lstm, attention])
    decoder_output = Dense(vocab_size, activation='softmax')(decoder_concat)
    
    model = Model(inputs=[image_input, decoder_input], outputs=decoder_output)
    return model

# Train model
def train_model(image_paths, padded_sequences, vocab_size, max_length, tokenizer, epochs=10, batch_size=32):
    image_data = load_and_preprocess_images(image_paths, (224, 224))
    
    model = define_model((224, 224, 3), vocab_size, max_length)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    model.fit([image_data, padded_sequences], padded_sequences, epochs=epochs, batch_size=batch_size)
    return model

# Train the model
trained_model = train_model(image_paths, padded_sequences, vocab_size, max_length, tokenizer)

# Evaluate model
def evaluate_model(model, images_data, tokenizer, max_length, image_directory):
    actual, predicted = list(), list()
    for image_data in images_data:
        image_path = os.path.join(image_directory, image_data['filename'])
        image = load_and_preprocess_images([image_path], (224, 224))
        image = np.expand_dims(image, axis=0)
        
        input_seq = np.zeros((1, max_length))
        input_seq[0, 0] = tokenizer.word_index['startseq']
        for i in range(1, max_length):
            predictions = model.predict([image, input_seq], verbose=0)
            predicted_word_index = np.argmax(predictions[0, i-1, :])
            input_seq[0, i] = predicted_word_index
            if tokenizer.index_word[predicted_word_index] == 'endseq':
                break
        
        actual_caption = []
        for word_index in padded_sequences[images.index(image_data['filename'])]:
            if word_index == 0:
                continue
            word = tokenizer.index_word[word_index]
            if word == 'endseq':
                break
            actual_caption.append(word)
        actual.append([actual_caption])
        
        predicted_caption = []
        for word_index in input_seq[0]:
            if word_index == 0:
                continue
            word = tokenizer.index_word[int(word_index)]
            if word == 'endseq':
                break
            predicted_caption.append(word)
        predicted.append(predicted_caption)
    
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

evaluate_model(trained_model, images_data, tokenizer, max_length, image_directory)
