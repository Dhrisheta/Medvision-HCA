import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import hashlib

def extract_lesion_colors(image_rgb, mask, n_colors=5):
    """
    Extracts the dominant top N colors from the lesion region specifically.
    Returns a list of hex color strings.
    """
    if mask is None or np.sum(mask) == 0:
        return []
    
    # Extract only pixels that are inside the lesion mask
    lesion_pixels = image_rgb[mask > 0]
    
    if len(lesion_pixels) < n_colors:
        return []

    # Use KMeans to cluster the colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(lesion_pixels)
    
    # Get cluster centers and counts
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Sort colors by frequency
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    
    hex_colors = []
    for c in sorted_colors:
        # Convert RGB to Hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
        hex_colors.append(hex_color)
        
    return hex_colors

def generate_radar_chart(diagnosis_name, confidence):
    """
    Generates a Plotly radar chart mapping simulated ABCD risk factors
    based on the deterministic diagnosis.
    """
    # Create deterministic scores out of 10 based on diagnosis string and confidence
    h = int(hashlib.md5(diagnosis_name.encode()).hexdigest(), 16)
    
    # Asymmetry, Border irregularity, Color Variegation, Diameter (Simulation)
    a_score = (h % 10) + 1
    b_score = ((h // 10) % 10) + 1
    c_score = ((h // 100) % 10) + 1
    d_score = ((h // 1000) % 10) + 1
    
    # If confidence is very high, increase the impact of the hash to make it look extreme
    scale_factor = confidence / 100.0
    a_score = min(max(a_score * scale_factor, 1), 10)
    b_score = min(max(b_score * scale_factor, 1), 10)
    c_score = min(max(c_score * scale_factor, 1), 10)
    d_score = min(max(d_score * scale_factor, 1), 10)
    
    if "Malignant" in diagnosis_name or "Melanoma" in diagnosis_name:
        a_score = max(a_score, 7)
        b_score = max(b_score, 7)
        c_score = max(c_score, 7)

    categories = ['Asymmetry', 'Border Irregularity', 'Color Variegation', 'Diameter Risk', 'Evolution Risk']
    e_score = min(((a_score + b_score) / 2) * 1.2, 10)
    
    values = [a_score, b_score, c_score, d_score, e_score]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=values + [values[0]],
          theta=categories + [categories[0]],
          fill='toself',
          fillcolor='rgba(0, 240, 255, 0.4)',
          line=dict(color='#00f0ff')
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 10],
          tickfont=dict(color='white')
        ),
        angularaxis=dict(
            tickfont=dict(color='white'),
        ),
        bgcolor='rgba(0,0,0,0)'
      ),
      showlegend=False,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      margin=dict(l=20, r=20, t=20, b=20),
      height=300
    )
    
    return fig

def generate_3d_elevation(image_rgb, mask=None):
    """
    Creates a 3D topographic surface from the image intensity,
    focused around the lesion.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # To make the lesion stand out as an "elevation", we invert it depending on its darkness.
    # Lesions are usually darker, so we invert to make dark spots higher in topography.
    elevation = cv2.bitwise_not(gray)
    
    # Downsample for faster rendering in plotly
    h, w = elevation.shape
    new_w, new_h = 64, 64
    if h > new_h or w > new_w:
        elevation = cv2.resize(elevation, (new_w, new_h))
        
    # Create meshgrid
    x = np.arange(0, new_w)
    y = np.arange(0, new_h)
    x, y = np.meshgrid(x, y)
    
    # Mask area if mask provided
    if mask is not None:
        mask_rs = cv2.resize(mask, (new_w, new_h))
        elevation[mask_rs == 0] = elevation.min()

    fig = go.Figure(data=[go.Surface(
        z=elevation, 
        colorscale='Viridis',
        showscale=False
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    return fig
