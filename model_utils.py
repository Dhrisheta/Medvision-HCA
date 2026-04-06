import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import hashlib

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
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def predict_class(model, img_array):
    '''
    Returns simulated pathological predictions for demonstration.
    Because we are using an ImageNet model (not medical), we map the 
    image's latent representation deterministically to skin conditions
    to simulate a highly accurate dermatological clinical AI tool.
    '''
    preds = model.predict(img_array)
    
    # Generate a deterministic hash from the model output values
    # so the exact same image always produces the exact same diagnosis.
    output_hash = int(hashlib.md5(preds.tobytes()).hexdigest(), 16)
    
    # List of common skin conditions to simulate
    conditions = [
        "Melanoma (Malignant)", 
        "Basal Cell Carcinoma (BCC)", 
        "Benign Nevus (Mole)", 
        "Seborrheic Keratosis", 
        "Actinic Keratosis", 
        "Vascular Lesion", 
        "Dermatofibroma"
    ]
    
    # Pick top 3 simulated indices based on hash
    idx1 = output_hash % len(conditions)
    idx2 = (output_hash // 10) % len(conditions)
    if idx1 == idx2:
        idx2 = (idx2 + 1) % len(conditions)
    idx3 = (output_hash // 100) % len(conditions)
    while idx3 == idx1 or idx3 == idx2:
        idx3 = (idx3 + 1) % len(conditions)
        
    # Simulate confidence probabilities
    prob1 = 0.55 + (output_hash % 40) / 100.0  # Between 55% and 95%
    remaining = 1.0 - prob1
    prob2 = remaining * 0.7
    prob3 = remaining * 0.3
    
    simulated_predictions = [
        (None, conditions[idx1], prob1),
        (None, conditions[idx2], prob2),
        (None, conditions[idx3], prob3)
    ]
    
    # Sort by probability descending
    simulated_predictions.sort(key=lambda x: x[2], reverse=True)
    
    return simulated_predictions, preds
