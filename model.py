import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Concatenate, GlobalAveragePooling2D, Dropout, Layer
from keras.models import Model
from keras.applications import ResNet50
from transformers import ViTModel, ViTImageProcessor
import torch
import numpy as np
from PIL import Image

# Define ViT feature extraction as a custom Keras layer
class ViTFeatureExtractor(Layer):
    def __init__(self, vit_model, vit_feature_extractor, **kwargs):
        super(ViTFeatureExtractor, self).__init__(**kwargs)
        self.vit_model = vit_model
        self.vit_feature_extractor = vit_feature_extractor

    def call(self, inputs):
        def extract_features(img):
            img_np = img.numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            vit_input_processed = self.vit_feature_extractor(images=img_pil, return_tensors="pt")
            self.vit_model.eval()
            with torch.no_grad():
                vit_output = self.vit_model(**vit_input_processed).last_hidden_state
            vit_output_flat = vit_output.view(vit_output.size(0), -1).numpy()
            return vit_output_flat
        
        vit_features = tf.py_function(func=extract_features, inp=[inputs], Tout=tf.float32)
        vit_features.set_shape([None, 151296])
        return vit_features

# Function to create the hybrid model
def create_hybrid_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    cnn_base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    cnn_output = GlobalAveragePooling2D()(cnn_base.output)

    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_features_layer = ViTFeatureExtractor(vit_model=vit_model, vit_feature_extractor=vit_feature_extractor)
    vit_features = vit_features_layer(inputs)

    vit_features_reduced = Dense(2048, activation='relu')(vit_features)
    fusion = Concatenate()([cnn_output, Flatten()(vit_features_reduced)])

    x = Dense(512, activation='relu')(fusion)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
