import cv2
import numpy as np

def calculate_damage_percentage(image_array):
    '''
    Analyzes the image array (RGB format from PIL) to compute the percentage
    of skin area that is affected by a lesion/abnormality.
    '''
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Segment Human Skin
    # Generic wide threshold for skin tones in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Total distinct skin pixels
    total_skin_pixels = cv2.countNonZero(skin_mask)
    if total_skin_pixels == 0:
        return 0.0, img_bgr, None # No skin detected
        
    # 2. Segment Lesion/Dark spots within the skin
    # Convert skin image to Grayscale to find dark anomalous regions
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding inverted to find darker spots (lesions)
    _, lesion_mask_rough = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # The lesion must be within the skin boundary
    lesion_mask_refined = cv2.bitwise_and(lesion_mask_rough, skin_mask)
    
    # Calculate lesion pixels
    lesion_pixels = cv2.countNonZero(lesion_mask_refined)
    
    # Compute Percentage
    damage_percentage = (lesion_pixels / total_skin_pixels) * 100.0
    
    # 3. Create Annotated Map (Green overlay for skin, Red for lesion contours)
    annotated_img = img_bgr.copy()
    
    # Find contours for drawing the exact borders
    contours, _ = cv2.findContours(lesion_mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw Lesion boundaries in striking RED
    cv2.drawContours(annotated_img, contours, -1, (0, 0, 255), 2)
    
    # Convert back to RGB for Streamlit displaying
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Clamp percentage between 0 and 100
    damage_percentage = min(max(damage_percentage, 0.0), 100.0)
    
    return damage_percentage, annotated_img_rgb, lesion_mask_refined
