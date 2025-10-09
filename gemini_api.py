import os
import json
import asyncio
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
 
load_dotenv()
 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
 
app = FastAPI(title="Gemini Text Processing API")
 
class TextInput(BaseModel):
    text: str
 
@app.post("/v1/summarize")
async def generate_summary(data: TextInput):
    try:
        prompt = f"Generate a summary of the following text in 2-3 lines:\n\n{data.text}"
 
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt            
        )
 
        summary = response.text.strip()
        return {"Summary": summary}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/v1/entities")
async def generate_entities(data: TextInput):
    try:
        prompt = f"""
        Extract and generate entities from the text below.
        Generate the entities in JSON format and include the fields: entity, type, and context.
        DO NOT include any markdown formatting (like ```json) or any surrounding text — only the raw JSON array.
 
        Text:
        {data.text}
        """
 
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt            
        )
 
        json_string = response.text.strip()
 
        entities_data = json.loads(json_string)
 
        return {"Entities": entities_data}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.post("/v1/few_shot")
async def few_shot_prompting(data: TextInput):
    prompt=f"{data.text}"
 
    response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt            
        )
 
    return response.text.strip()
 
@app.post("/v1/zero_shot")
async def few_shot_prompting(data: TextInput):
    prompt=f"{data.text}"
 
    response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt            
        )
 
    return response.text.strip()
 
@app.post("/v1/reasoning")
async def chain_of_thought(data:TextInput):
    prompt =f"{data.text}"
 
    response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt            
        )
 
    return response.text.strip()
 
@app.post("/v1/file_summary",response_class=PlainTextResponse)
async def generate_pdf_summary():
    doc_url = "https://arxiv.org/pdf/2203.02155"
    async with httpx.AsyncClient() as async_client:
        doc_response = await async_client.get(doc_url)
        doc_data = doc_response.content
 
    prompt = "Summarize the document"
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents = [
        types.Part.from_bytes(
            data=doc_data,
            mime_type='application/pdf',
        ),
        prompt])
   
    return response.text.strip()