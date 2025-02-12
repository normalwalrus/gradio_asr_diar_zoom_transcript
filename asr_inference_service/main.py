"""
API module for ASR service.

This module provides the FastAPI application for performing ASR.
"""

import io
import json
import logging
import os
import tempfile
import shutil
from typing import List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.status import HTTP_200_OK

from asr_inference_service.model import ASRModelForInference
from asr_inference_service.schemas import ASRResponse, HealthResponse, DenoiseResponse
from asr_inference_service.denoise import DENOISER
from utils.audio_preprocessing import resample_audio_array

SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8080

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

app = FastAPI()
model = ASRModelForInference(
    model_dir=os.environ["PRETRAINED_MODEL_DIR"],
    diar_dir=os.environ['PRETRAINED_DIAR_MODEL_DIR'],
    sample_rate=int(os.environ["SAMPLE_RATE"]),
    device=os.environ["DEVICE"],
    timestamp_format=os.environ['TIMESTAMPS_FORMAT'],
    min_segment_length=float(os.environ['MIN_SEGMENT_LENGTH']),
    min_silence_length=float(os.environ['MIN_SILENCE_LENGTH'])
)

if int(os.environ['DENOISER']):
    denoiser = DENOISER(
        device=os.environ["DEVICE"],
        dry=float(os.environ["DRY"]),
        amplification_factor=float(os.environ["AMPLIFICATION_FACTOR"])
    )
    
SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])

class AudioData(BaseModel):
    array: list


@app.get("/", status_code=HTTP_200_OK)
async def read_root():
    """Root Call"""
    return {"message": "This is an ASR service."}


@app.get("/health")
async def read_health() -> HealthResponse:
    """
    Check if the API endpoint is available.

    This endpoint is used by Docker to check the health of the container.
    """
    return {"status": "HEALTHY"}


@app.post("/v1/transcribe", response_model=ASRResponse)
async def transcribe(data: Request):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    data = await data.json()

    transcription = model.infer(data["array"], 16000)

    return {"transcription": str(transcription)}


@app.post("/v1/transcribe_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")

    # Receive the audio bytes from the request
    audio_bytes = file.file.read()

    # load with soundfile, data will be a numpy array
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    transcription = model.infer(data, samplerate)

    return {"transcription": str(transcription)}

@app.post("/v1/denoise_filepath", response_model=DenoiseResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        
        shutil.copyfileobj(file.file, temp_file)
        
        temp_file_path = temp_file.name
        
        denoised = denoiser.denoise(temp_file_path)

    return {"denoise_audio": denoised.tolist()}

@app.post("/v1/transcribe_diarize_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, saves it as a temp .wav file and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")
    
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        # Write the content of the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        
        temp_file_path = temp_file.name
        transcription = model.diar_inference(temp_file_path)

    return {"transcription": str(transcription)}

@app.post("/v1/transcribe_diarize_denoise_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, saves it as a temp .wav file and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")
    
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        # Write the content of the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        
        temp_file_path = temp_file.name
        
        denoised = denoiser.denoise(temp_file_path)
    
        sf.write(temp_file_path, denoised, int(os.environ["SAMPLE_RATE"]))
        
        transcription = model.diar_inference(temp_file_path)

    return {"transcription": str(transcription)}

@app.post("/v1/transcribe_resample_diarize_filepath", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    """Function call to takes in an audio file as bytes, saves it as a temp .wav file and executes model inference"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File uploaded is not a wav file.")
    
    audio_bytes = file.file.read()
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    y = resample_audio_array(data, samplerate, SAMPLE_RATE)
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        
        sf.write(temp_file, y, SAMPLE_RATE)
        temp_file_path = temp_file.name
        transcription = model.diar_inference(temp_file_path)

    return {"transcription": str(transcription)}


def start():
    """Launched with `start` at root level"""
    uvicorn.run(
        "asr_inference_service.main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=False,
    )


if __name__ == "__main__":
    start()
