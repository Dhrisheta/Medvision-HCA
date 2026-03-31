import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    '''
    Generates a Grad-CAM heatmap for the given image and model.
    '''
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # Relu to keep only positive influence (discarding negative correlations)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img_path_or_array, heatmap, alpha=0.4):
    '''
    Overlays the computed Grad-CAM heatmap on the original image.
    '''
    if isinstance(img_path_or_array, str):
        img = tf.keras.preprocessing.image.load_img(img_path_or_array)
        img = tf.keras.preprocessing.image.img_to_array(img)
    else:
        # Assuming numpy array input from PIL Image
        img = np.array(img_path_or_array)

    # Convert RGB/RGBA to RGB
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert from BGR to RGB (OpenCV default is BGR)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image dimensions
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = np.uint8(img * (1 - alpha) + jet * alpha)
    return superimposed_img
