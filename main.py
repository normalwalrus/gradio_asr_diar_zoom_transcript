import os
import tempfile
from datetime import datetime

import gradio as gr
import librosa
import soundfile as sf

from asr_inference_service.model import ASRModelForInference
from utils.audio_preprocessing import resample_audio_array
from utils.utils import (
    convert_diar_string_to_list,
    convert_list_of_timestamps_to_seconds,
    get_most_frequent_speaker,
    get_speakers_names,
    get_timestamps_for_speaker,
    replacement_of_string_in_text,
    get_timestamps_for_speaker_timestamps
)

model = ASRModelForInference(
    model_dir=os.environ["PRETRAINED_MODEL_DIR"],
    sample_rate=int(os.environ["SAMPLE_RATE"]),
    device=os.environ["DEVICE"],
    timestamp_format=os.environ["TIMESTAMPS_FORMAT"],
    min_segment_length=float(os.environ["MIN_SEGMENT_LENGTH"]),
    min_silence_length=float(os.environ["MIN_SILENCE_LENGTH"]),
)

SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])

default_download_button = gr.DownloadButton(label="Load the .txt file to download", value=None)

TITLE = '''# DH Transcription Service'''

def timestamp_logic(file_input, speaker: str):
    """
    Handles start and end timestamp finding for specific speaker
    """

    print(file_input)
    print(speaker)

    start, end, _ = get_timestamps_for_speaker_timestamps(speaker, file_input)

    final_string = (
        f"Start time for interviewee: {start} \nEnd time for interviewee: {end}"
    )

    return final_string


def transcription_logic(audio_filepath, file_input=None, speaker=None, offset_sec=1.5, end_offset_sec=240):
    """
    Overall Transcription logic, chaining all functionalities tgt:

    1. Loads audio in and resamples
    2. Handles if there is a specific speaker to focus on
    3. Handles diarization and transcription calls to the model
    4. Returns the transcription

    """
    
    if audio_filepath == None:
        return

    data, samplerate = librosa.load(audio_filepath)

    y = resample_audio_array(data, samplerate, SAMPLE_RATE)

    if speaker:
        # If Zoom Transcript is given

        matches = (
            get_timestamps_for_speaker(file_input)
        )
        
        start, end, _ = get_timestamps_for_speaker_timestamps(speaker, file_input)
        
        start_time = datetime.strptime(start, "%H:%M:%S.%f")
        end_time = datetime.strptime(end, "%H:%M:%S.%f")

        # Convert to total seconds
        start_seconds = (
            start_time.hour * 3600
            + start_time.minute * 60
            + start_time.second
            + start_time.microsecond / 1_000_000
        )
        end_seconds = (
            end_time.hour * 3600
            + end_time.minute * 60
            + end_time.second
            + start_time.microsecond / 1_000_000
        )

        if start_seconds - offset_sec > 0:

            start_seconds = start_seconds - offset_sec

        start_timeframe, end_timeframe = (
            start_seconds * SAMPLE_RATE,
            end_seconds * SAMPLE_RATE,
        )
        
        if end_timeframe + end_offset_sec * SAMPLE_RATE <= len(y):
            
            end_timeframe += end_offset_sec * SAMPLE_RATE
        
        else:
            
            end_timeframe = len(y)

        print(f"Start timeframes ({SAMPLE_RATE}Hz) : {start_timeframe}")
        print(f"End timeframes ({SAMPLE_RATE}Hz) : {end_timeframe}")

        truncated_audio_array = y[int(start_timeframe) : int(end_timeframe)]
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:

            sf.write(temp_file, truncated_audio_array, SAMPLE_RATE)
            temp_file_path = temp_file.name

            transcription = model.diar_inference(temp_file_path)

        average_actual_time_sec = convert_list_of_timestamps_to_seconds(
            matches
        )
        transcription_time_segments = convert_diar_string_to_list(
            transcription, start_seconds
        )

        speaker_chosen_list = get_most_frequent_speaker(
            transcription_time_segments, average_actual_time_sec
        )
        
        for speaker_chosen in speaker_chosen_list:

            transcription = replacement_of_string_in_text(
                transcription, speaker_chosen, speaker_chosen_list[speaker_chosen]
            )
            
        transcription = (
            f"Transcriptions for {speaker} as interviewee: \n\n" + transcription
        )

    else:
        # If Zoom transcript is not given, just dairization and transcription

        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:

            sf.write(temp_file, y, SAMPLE_RATE)
            temp_file_path = temp_file.name
            transcription = model.diar_inference(temp_file_path)

    return transcription


def download_logic(transcription, speaker_choice = None, download_button = gr.DownloadButton()):
    """
    Download logic to download the transcript
    """
    print(f"Speaker choice for downloading: {speaker_choice}")
    
    if transcription == None or len(transcription) == 0:
        return default_download_button
    
    if speaker_choice:
        output_filepath = f"interview_{speaker_choice}_transcription.txt"
    else:
        output_filepath = f"transcription.txt"

    with open(output_filepath, "w") as text_file:
        text_file.write(transcription)
    
    if download_button:
        download_button = default_download_button
    else:
        download_button = gr.DownloadButton(label=f"Download {output_filepath}", value=output_filepath)
    
    return download_button

def reset_download_button():
    '''
    When audio file is uploaded, reset the download button
    '''
    return default_download_button


"""
Overall Gradio app
"""
with gr.Blocks() as demo:
    
    with gr.Row():
        
        gr.Markdown(TITLE)

    with gr.Row():

        with gr.Column(0):

            audio_input = gr.Audio(label="Audio Clip", type="filepath")

            file_input = gr.File(label="Zoom Transcript")
            speaker_choice = gr.Radio([], label="Choose Interviewee: ")
            
            transcribe_button = gr.Button("Start Transcription!")

            file_input.change(get_speakers_names, file_input, speaker_choice)

        with gr.Column(1):

            transcript_outputs = [gr.Textbox(label="Transcription")]
            timestamps_outputs = [gr.Textbox(label="Interview Start and End timestamp")]
            summarization_outputs = [gr.Textbox(label="Summarization")]

            speaker_choice.input(
                timestamp_logic, [file_input, speaker_choice], timestamps_outputs
            )
            
            transcribe_button.click(
                transcription_logic,
                [audio_input, file_input, speaker_choice],
                transcript_outputs,
            )
            
            download_button = gr.DownloadButton(label="Load the .txt file to download", value=None)
            download_button.click(download_logic, [transcript_outputs[0], speaker_choice, download_button], download_button)
            
            audio_input.upload(
                reset_download_button,
                None,
                download_button
            )


if __name__ == "__main__":

    demo.launch()
