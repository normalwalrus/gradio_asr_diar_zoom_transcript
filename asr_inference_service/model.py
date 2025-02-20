"""ASR Inference Model Class"""

import logging
import os
from time import perf_counter

import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from asr_inference_service.diarizer import PyannoteDiarizer

class ASRModelForInference:
    """Base class for ASR model for inference"""

    def __init__(self, model_dir: str,
                 sample_rate: int = 16000,
                 device: str = 'cpu',
                 timestamp_format: str = 'seconds',
                 min_segment_length = 0.5,
                 min_silence_length = 0):
        """
        Inputs:
            model_dir (str): path to model directory
            sample_rate (int): the target sample rate in which the model accepts
        """
        
        device = device if device in ['cuda', 'cpu'] else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_number = [0] if device == 'cuda' else 1
        self.accelerator = 'gpu' if device == 'cuda' else 'cpu'
        
        self.init_model(model_dir, device, min_segment_length, min_silence_length)
        self.timestamp_format = timestamp_format if timestamp_format in ['minutes', 'seconds'] else 'seconds'
        logging.info("Running on device: %s", device)
        self.target_sr = sample_rate

    def init_model(self,
                   model_dir: str,
                   device: str,
                   min_segment_length: float,
                   min_silence_length: float):
        """Method to initialise model on class initialisation

        Inputs:
            model_dir (str): path to model directory
        """
        logging.info("Loading model...")
        model_load_start = perf_counter()
        
        # Instantiating Diarizer
        # self.diar_model = NemoDiarizer(self.diar_dir, device=self.device_number, accelerator=self.accelerator)
        self.diar_model = PyannoteDiarizer(device = device, 
                                           min_segment_length=min_segment_length,
                                           min_silence_length=min_silence_length)

        self.device = device
        self.torch_dtype = torch.float16 if self.device=='cuda' else torch.float32
        logging.info("Torch dtype: %s", self.torch_dtype)
        
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
        self.model.to(device)
        self.model.config.forced_decoder_ids = None
        self.model.eval()

        #################### Set to English and Transcription task ###############
        self.language = "English"
        self.task = "transcribe"
        self.model.config.forced_decoder_ids = (
            self.processor.tokenizer.get_decoder_prompt_ids(
                language=self.language, task=self.task
            )
        )
        self.model.config.suppress_tokens = []
        self.model.generation_config.forced_decoder_ids = (
            self.processor.tokenizer.get_decoder_prompt_ids(
                language=self.language, task=self.task
            )
        )
        self.model.generation_config.suppress_tokens = []
        ##########################################################################

        model_load_end = perf_counter()
        logging.info(
            "Models loaded. Elapsed time: %s", model_load_end - model_load_start
        )

    def load_audio(self, audio_filepath: str) -> np.ndarray:
        """Method to load an audio filepath to generate a waveform, it automatically
        standardises the waveform to the target sample rate and channel

        Inputs:
            audio_filepath (str): path to the audio file

        Returns:
            waveform (np.ndarray) of shape (T,)
        """

        waveform, _ = librosa.load(audio_filepath, sr=self.target_sr, mono=True)

        return waveform

    def infer(self, waveform: np.ndarray, input_sr: int) -> str:
        """Method to run inference on a waveform to generate a transcription

        Inputs:
            waveform (np.ndarray): Takes in waveform of shape (T,)
            input_sr (int): Sample rate of input waveform

        Returns:
            transcription (str): Output text generated by the ASR model
        """
        inference_start = perf_counter()

        if input_sr != self.target_sr:
            waveform = librosa.resample(
                waveform, orig_sr=input_sr, target_sr=self.target_sr
            )
            
        if len(waveform.shape) == 2:
            logging.info("Converting Steoreo Waveform to Mono Waveform")
            waveform = waveform.mean(axis=1)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        transcription = pipe(np.array(waveform))
        inference_end = perf_counter()
        logging.info(
            "Inference Model triggered. Elapsed time: %s",
            inference_end - inference_start,
        )

        return transcription["text"]

    def diar_inference(self, filepath: str):
        """Method to call vad methods and using segments of speech to transcribe using the infer method

        Inputs:
            waveform (np.ndarray): Takes in waveform of shape (T,)
            input_sr (int): Sample rate of input waveform

        Returns:
            final_transcription (str): transcription with timestamps attached to it
        """
        diarizer_start = perf_counter()
        logging.info(
            "Diarization Model triggered."
        )
        
        segments = self.diar_model.diarize(filepath)
        waveform = self.load_audio(filepath)
        
        diarizer_end = perf_counter()
        logging.info(
            "Diarization Model Done. Elapsed time: %s",
            diarizer_end - diarizer_start,
        )
        
        final_transcription=""
        
        for x in range(len(segments)):
            start_time = segments["start_time"][x]
            end_time = segments["end_time"][x]
            
            start_frame = int(start_time * self.target_sr)
            end_frame = int(end_time * self.target_sr)

            split_audio = waveform[start_frame:end_frame]

            transcription = self.infer(split_audio, self.target_sr)
            
            if self.timestamp_format == 'minutes':
                start_time = start_time/60
                end_time = end_time/60
                
            segment_string = f"[{start_time:.2f} - {end_time:.2f}] [{segments['speaker'][x]}] : {transcription}\n\n"
            final_transcription = "".join([final_transcription, segment_string])
        
        return final_transcription


if __name__ == "__main__":
    pass