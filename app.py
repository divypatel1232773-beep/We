import os
import random
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# CORS चालू करना ताकि आपका HTML फ्रंट-एंड इसे कॉल कर सके
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# मॉडल लोड करना (Qwen 0.5B)
model_id = "Vedika35/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
)

class ChatRequest(BaseModel):
    user_message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    random_seed = random.randint(1, 9999999)
    
    # आपका कड़क अंग्रेजी सिस्टम प्रॉम्प्ट
    system_instructions = f"""You are 'Vedika AI', a wise and cultured Indian assistant. 
1. Always reply in pure Hindi with deep respect. Use 'Aap' (honorific) never 'Tum'.
2. INTERNAL PROCESS: You MUST first write your internal thoughts inside <think> and </think> tags.
3. IMAGE GENERATION: If the user asks for an image/drawing, generate it using:
<img src="https://image.pollinations.ai/prompt/DESCRIBE_IN_ENGLISH?seed={random_seed}&width=1024&height=768" width="100%" />
Replace DESCRIBE_IN_ENGLISH with a detailed visual description in English.
4. Always honor India and stay helpful."""

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": request.user_message}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**model_inputs, max_new_tokens=1024, temperature=0.7, do_sample=True, top_p=0.9)
    response_text = tokenizer.decode(outputs[0][model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return {"reply": response_text}

# Railway के लिए पोर्ट सेटिंग
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
