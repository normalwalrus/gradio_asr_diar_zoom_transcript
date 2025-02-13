# Gradio Transcription-Diarization App with Zoom Transcript intergration

## About

This is a gradio app that performs diarisation and transcription on an audio clip.
If the audio clip is taken from a zoom recording, you can upload the transcript and select individual people to transcribe for the duration they speak.

Models used:

1. Transcription: Whisper-v3
2. Diarization: Pyannote Diarizer 3.1

Notes:

1. All audio clips are resampled to 16kHz
2. All audio clips are rechanneled to mono

## Setting it up

### Pre-setup

1. Create a /pretrained_models folder in the main directory and download the Whisper-v3 model into it ( from https://huggingface.co/openai/whisper-large-v3 )
2. Input your huggingface token into:

   RUN ["python", "-c", "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token='HF_TOKEN_HERE')"]

   This is found in the last few lines of the docker file

### Docker Commands

1. Build the Dockerfile using

```
docker-compose build asr-hr-service
```

2. Spin up the Docker container

```
docker-compose up asr-hr-service
```

3. Access the Gradio App at http://localhost:7860/

## Workflow

### With Zoom Transcript

1. Upload the Zoom transcript under the “Zoom Transcript” Section
2. Choose the interviewee to focus on
3. Upload the Audio clip that corresponds to the Zoom Transcript under the “Audio Clip” Section
4. Press the “Start Transcription!” Button
5. Press the Download button to Download the transcription\_{speaker}.txt

### Without Zoom Transcript

1. Upload the Audio clip under the “Audio Clip” Section
2. Press the “Start Transcription!” Button
3. Press the Download button to Download the transcription.txt
