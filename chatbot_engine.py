import os
import hashlib
import base64
from io import BytesIO

def _pil_to_base64(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_chat_response(prompt, image_pil, api_key, history, current_diagnosis):
    """
    Generates a response from the AI assistant. If API key is provided, uses xAI API with grok-2-vision-1212.
    Otherwise, simulates a response based on the deterministic diagnosis.
    """
    if not api_key:
        return _simulate_response(prompt, current_diagnosis)
        
    try:
        from openai import OpenAI
        # Initialize xAI client via OpenAI SDK
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        
        system_instructions = (
            "You are an advanced digital dermatologist assistant. "
            f"The current visual screening has suggested a preliminary simulated diagnosis of {current_diagnosis}. "
            "Use this as context but rely on your own vision capabilities to analyze the provided image directly. "
            "Maintain a highly professional, clinical, and helpful tone. Always state that this is an AI tool and not a replacement for a doctor."
        )
        
        # Convert PIL Image to Base64
        base64_image = _pil_to_base64(image_pil)
        
        constructed_messages = [{"role": "system", "content": system_instructions}]
        
        # Append previous conversation history
        # We assume history contains [{"role": "user"/"assistant", "content": "..."}]
        # The history passed includes all previous messages but NOT the current prompt (which hasn't been added yet in app.py logic)
        for msg in history:
            constructed_messages.append({"role": msg["role"], "content": msg["content"]})
                 
        # Append the current prompt alongside the image
        constructed_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        })
        
        response = client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=constructed_messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"System Malfunction (API Error): {str(e)}. Please check your API key and network connection."

def _simulate_response(prompt, current_diagnosis):
    """Fallback simulated response generator"""
    prompt_lower = prompt.lower()
    
    responses = [
        "Based on the visual patterns, I notice irregularities that align with ",
        "My analysis of the color distribution and borders points towards ",
        "The focal characteristics in the dermoscopic simulation suggest features of "
    ]
    
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    prefix = responses[h % len(responses)]
    
    if "treatment" in prompt_lower or "cure" in prompt_lower:
        return f"Treatment for {current_diagnosis} heavily depends on professional medical evaluation. While my simulation indicates it might be {current_diagnosis}, you MUST consult a human dermatologist for surgical excision, topical therapies, or cryotherapy guidance."
    if "what is" in prompt_lower or "explain" in prompt_lower:
        return f"{current_diagnosis} is a categorized skin condition. My topographical scanners flagged this due to specific structural asymmetries and border abnormalities. Please refer to clinical guidelines for formal definitions."
    if "danger" in prompt_lower or "risk" in prompt_lower or "fatal" in prompt_lower:
        if "Malignant" in current_diagnosis or "Melanoma" in current_diagnosis:
            return f"The simulated diagnosis is {current_diagnosis}, which typically carries a HIGH risk profile. Please seek immediate professional medical attention."
        else:
            return f"The simulated diagnosis is {current_diagnosis}. While it may have a lower immediate risk profile compared to malignant conditions, any changing lesion should be examined by a doctor."
            
    return f"{prefix} {current_diagnosis}. Please note that I am currently operating in MOCK SIMULATION MODE without an active API connection to the central neural core. Therefore my responses are deterministic demonstrations."
