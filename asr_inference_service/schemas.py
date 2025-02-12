"""Schemas for the API service"""

from pydantic import BaseModel


# pylint: disable=too-few-public-methods
class ASRResponse(BaseModel):
    """ASR service response format

    Attributes:
        status_code (int): default success status code
        transcription (str): output transcription from the ASR model
    """

    status_code: int = 200
    transcription: str

class DenoiseResponse(BaseModel):
    """Denoiser service response format

    Attributes:
        status_code (int): default success status code
        transcription (str): output transcription from the ASR model
    """

    status_code: int = 200
    denoise_audio: list

# pylint: disable=too-few-public-methods
class HealthResponse(BaseModel):
    """
    Response model for the `health` API.

    Attributes:
        status (str): The status message.
    """

    status: str = "HEALTHY"
