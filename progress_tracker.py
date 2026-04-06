import numpy as np
import cv2
import plotly.graph_objects as go
import hashlib
from datetime import datetime, timedelta

def generate_risk_meter_chart(diagnosis_name, confidence, damage_pct):
    """
    Generates a Plotly Gauge Chart for the 'Is This Serious?' Risk Meter.
    🟢 Low risk
    🟡 Moderate
    🔴 High (See doctor immediately)
    """
    # Deterministic risk scale 0-100
    h = int(hashlib.md5(diagnosis_name.encode()).hexdigest(), 16)
    base_risk = (h % 50)
    
    if "Malignant" in diagnosis_name or "Melanoma" in diagnosis_name:
        base_risk += 50 # Force high risk
    elif "Carcinoma" in diagnosis_name:
        base_risk += 40
        
    risk_score = min(base_risk + (damage_pct * 0.5), 100)
    
    # Determine zone text
    if risk_score > 70:
        zone_color = "red"
        title = "🔴 HIGH RISK (Consult Doctor)"
    elif risk_score > 35:
        zone_color = "yellow"
        title = "🟡 MODERATE RISK (Monitor)"
    else:
        zone_color = "green"
        title = "🟢 LOW RISK (Benign Pattern)"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': "rgba(0, 255, 0, 0.4)"},
                {'range': [35, 70], 'color': "rgba(255, 255, 0, 0.4)"},
                {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.4)"}],
            'threshold': {
                'line': {'color': zone_color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score}
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': 'white'}
    )
    return fig, risk_score

def generate_severity_chart(current_damage_pct):
    """
    Generates a mock timeline of the severity leading up to today based on current percentage.
    """
    dates = []
    values = []
    
    # Generate past 6 months
    today = datetime.now()
    h = int(hashlib.md5(str(current_damage_pct).encode()).hexdigest(), 16)
    is_worsening = (h % 2 == 0)
    
    for i in range(5, -1, -1):
        dt = today - timedelta(days=30*i)
        dates.append(dt.strftime("%b %Y"))
        
        if is_worsening:
            # Started lower, got to current
            val = current_damage_pct * (1.0 - (0.15 * i))
        else:
            # Started higher, got to current (improving)
            val = current_damage_pct * (1.0 + (0.15 * i))
            
        values.append(max(val, 0.5)) # Minimum 0.5%
        
    values[-1] = current_damage_pct # Ensure exact match for today
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        name='Severity',
        line=dict(color='#ff3366' if is_worsening else '#00f0ff', width=3),
        marker=dict(size=8),
        fill='tozeroy', 
        fillcolor='rgba(255, 51, 102, 0.2)' if is_worsening else 'rgba(0, 240, 255, 0.2)'
    ))
    
    fig.update_layout(
        title="Progression Tracker (6 Months)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': 'white'},
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="white")
    )
    
    return fig

def simulate_past_image(image_rgb):
    """
    Applies Gaussian Blur and color shift to simulate an older, smaller lesion state.
    """
    # Just blur slightly and reduce contrast to simulate "before"
    img_blur = cv2.GaussianBlur(image_rgb, (15, 15), 0)
    img_past = cv2.convertScaleAbs(img_blur, alpha=0.9, beta=0)
    return img_past
