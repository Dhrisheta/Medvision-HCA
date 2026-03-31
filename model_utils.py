import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

def load_model():
    '''
    Loads a pre-trained MobileNetV2 model.
    Using ImageNet weights for demonstration. For a real medical app,
    this should load a custom-trained model (.h5 or SavedModel)
    specifically designed for identifying pathologies (e.g. skin lesions).
    '''
    model = MobileNetV2(weights='imagenet')
    return model

def prepare_image(img_pil):
    '''
    Prepares a PIL image for prediction with MobileNetV2.
    '''
    img_resized = img_pil.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply mobilenet_v2 specific preprocessing
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def predict_class(model, img_array):
    '''
    Returns top 3 predictions for the image.
    '''
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded, preds
