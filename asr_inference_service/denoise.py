import torch
import torchaudio
import logging
import soundfile as sf

from denoiser import pretrained
from denoiser.dsp import convert_audio
from time import perf_counter


logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


class DENOISER:
    """Base class for denoising model"""
    
    def __init__(self, 
                 device: str, 
                 dry: float, 
                 amplification_factor: float = 1) -> None:
        """Method to initialise denoiser class initialisation

        Inputs:
            device (str): path to model directory
            dry (float): value from 1 to 0, with 0 being the strongest denoiser
            amplification_factor (float): used for amplifying the audio clip (choose 1 to let waveform be unchanged)
        """
        logging.info("Denoiser loading... ")
        denoiser_load_start = perf_counter()
        
        self.device = device if device in ['cuda', 'cpu'] else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            self.model = pretrained.dns64().cuda()
        else:
            self.model = pretrained.dns64().cpu()
            
        self.dry = dry
        self.amplification_factor = amplification_factor
        
        logging.info(
            "Denoiser Dry: %s",
            self.dry,
        )
        
        denoiser_load_end = perf_counter()
        logging.info(
            "Denoiser loaded. Elapsed time: %s", denoiser_load_end - denoiser_load_start
        )
        
    
    def denoise(self, input_audio_filepath: str):
        """
        Method to run denoising on an audiofile to generate a numpy array of denoised audio

        Inputs:
            input_audio_filepath (string): Takes in filepath of the input audio
        
        Returns:
            denosied (numpy.ndarray): Output numpy array with denoised audio
        """
        
        logging.info("Denoiser triggered.")
        wav, sr = torchaudio.load(input_audio_filepath)
        
        wav = self.amplify_audio(wav=wav, amplification_factor=self.amplification_factor)
        
        if self.device == 'cuda':
            wav = convert_audio(wav.cuda(), sr, self.model.sample_rate, self.model.chin)
        else:
            wav = convert_audio(wav, sr, self.model.sample_rate, self.model.chin)
        
        with torch.no_grad():
            denoised = self.model(wav[None])
            denoised = (1 - self.dry) * denoised + self.dry * wav[None]

        denoised = denoised[0]
        denoised = denoised.data.cpu().numpy()[0]
        logging.info("Denoiser Complete.")
        
        return denoised
    
    def amplify_audio(self, wav, amplification_factor):
        """
        Method to amplify an audio tensor / numpy array by an amplification factor

        Inputs:
            wav (torch.tensor/numpy.ndarray): audio array/tensor to be amplified
            amplification_factor (float): the amount to amplify the audio by
        
        Returns:
            amplified_wav (torch.tensor/numpy.ndarray): Output tensor/array that has been amplified
        """
        
        logging.info("Amplification Triggered.")
        amplified_wav = wav * amplification_factor
        amplified_wav = torch.clamp(amplified_wav, min=-1.0, max=1.0)
        logging.info("Amplification Complete")
        
        return amplified_wav