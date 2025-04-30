##Vibecoded eng ver that returns audio from http://127.0.0.1:8000/synthesize as binary blob
#https://github.com/index-tts/index-tts/pull/131
#orig auth https://github.com/itltf512116
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse, Response
import os
import time
import uvicorn
from indextts.infer import IndexTTS
import tempfile
import hashlib
from typing import Dict
import base64
from pydantic import BaseModel

app = FastAPI(title="IndexTTS API")

# Initialize the TTS model
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

# Ensure the 'prompts' directory exists
os.makedirs("prompts", exist_ok=True)

@app.post("/upload_audio")
async def upload_audio(audio: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload an audio file and save it
   
    Parameters:
    - audio: The audio file
   
    Returns:
    - A JSON response containing the saved filename
    """
    # Read the file content and calculate its MD5 hash
    content = await audio.read()
    md5_hash = hashlib.md5(content).hexdigest()
   
    # Get the file extension
    file_extension = os.path.splitext(audio.filename)[1]
    if not file_extension:
        file_extension = ".wav"  # Default extension
   
    # Generate a new filename
    new_filename = f"{md5_hash}{file_extension}"
    save_path = os.path.join("prompts", new_filename)
   
    # Save the file
    with open(save_path, "wb") as f:
        f.write(content)
   
    return {"filename": new_filename}

@app.post("/synthesize")
async def synthesize_speech(
    prompt_audio: UploadFile = File(...),
    text: str = Form(...),
    infer_mode: str = Form("普通推理")
):
    """
    Synthesize speech API
   
    Parameters:
    - prompt_audio: Reference audio file
    - text: Text to synthesize
    - infer_mode: Inference mode ("普通推理" or "批次推理")
   
    Returns:
    -  Audio file as a binary blob
    """
    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await prompt_audio.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name
   
    # Generate the output file path
    output_path = os.path.join("outputs", f"api_synth_{int(time.time())}.wav")
   
    try:
        # Call the TTS model for synthesis
        if infer_mode == "普通推理":
            output = tts.infer(temp_audio_path, text, output_path)
        else:
            output = tts.infer_fast(temp_audio_path, text, output_path)
       
        # Read the generated audio file
        with open(output, "rb") as audio_file:
            audio_data = audio_file.read()
       
        # Return the audio data as a binary blob with appropriate content type
        return Response(content=audio_data, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "message": f"Synthesis failed: {str(e)}"
            }
        )
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)

class SynthesizeRequest(BaseModel):
    filename: str
    text: str
    infer_mode: str = "普通推理"

@app.post("/synthesize_by_filename")
async def synthesize_speech_by_filename(request: SynthesizeRequest = Body(...)):
    """
    Synthesize speech using an uploaded audio file
   
    Parameters:
    - request: Request body containing filename, text, and infer_mode
   
    Returns:
    - Audio file as binary blob
    """
    # Construct the full path to the audio file
    prompt_audio_path = os.path.join("prompts", request.filename)
   
    # Check if the file exists
    if not os.path.exists(prompt_audio_path):
        return JSONResponse(
            status_code=404,
            content={
                "code": 2,
                "message": f"Audio file {request.filename} does not exist"
            }
        )
   
    # Generate the output file path
    output_path = os.path.join("outputs", f"api_synth_{int(time.time())}.wav")
   
    try:
        # Call the TTS model for synthesis
        if request.infer_mode == "普通推理":
            output = tts.infer(prompt_audio_path, request.text, output_path)
        else:
            output = tts.infer_fast(prompt_audio_path, request.text, output_path)
       
        # Read the generated audio file
        with open(output, "rb") as audio_file:
            audio_data = audio_file.read()
       
        # Return the audio data as a binary blob
        return Response(content=audio_data, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "code": 1,
                "message": f"Synthesis failed: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
