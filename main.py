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
    download_string_as_txt
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


def timestamp_logic(file_input, speaker: str):
    """
    Handles start and end timestamp finding for specific speaker
    """

    print(file_input)
    print(speaker)

    start, end, _ = get_timestamps_for_speaker(speaker, file_input)

    final_string = (
        f"Start time for interviewee: {start} \nEnd time for interviewee: {end}"
    )

    return final_string


def transcription_logic(audio_filepath, file_input=None, speaker=None, offset_sec=1.5):
    """
    Overall Transcription logic, chaining all functionalities tgt:

    1. Loads audio in and resamples
    2. Handles if there is a specific speaker to focus on
    3. Handles diarization and transcription calls to the model
    4. Returns the transcription

    """

    data, samplerate = librosa.load(audio_filepath)

    y = resample_audio_array(data, samplerate, SAMPLE_RATE)

    if speaker:
        # If Zoom Transcript is given

        start_timestamps, end_timestamps, matches_timestamps = (
            get_timestamps_for_speaker(speaker, file_input)
        )
        start_time = datetime.strptime(start_timestamps, "%H:%M:%S.%f")
        end_time = datetime.strptime(end_timestamps, "%H:%M:%S.%f")

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

        print(f"Start timeframes ({SAMPLE_RATE}Hz) : {start_timeframe}")
        print(f"End timeframes ({SAMPLE_RATE}Hz) : {end_timeframe}")

        truncated_audio_array = y[int(start_timeframe) : int(end_timeframe)]
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:

            sf.write(temp_file, truncated_audio_array, SAMPLE_RATE)
            temp_file_path = temp_file.name

            transcription = model.diar_inference(temp_file_path)

        average_actual_time_sec = convert_list_of_timestamps_to_seconds(
            matches_timestamps
        )
        transcription_time_segments = convert_diar_string_to_list(
            transcription, start_seconds
        )

        speaker_chosen = get_most_frequent_speaker(
            transcription_time_segments, average_actual_time_sec
        )

        transcription = replacement_of_string_in_text(
            transcription, speaker_chosen, speaker
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


def download_logic(transcription):
    """
    Download logic to download the transcript
    """

    if transcription == None:
        return

    output_filepath = f"transcription.txt"

    with open(output_filepath, "w") as text_file:
        text_file.write(transcription)

    return output_filepath


"""
Overall Gradio app
"""
with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column(0):

            audio_input = gr.Audio(label="Audio Clip", type="filepath")

            file_input = gr.File(label="Zoom Transcript")
            speaker_choice = gr.Radio([], label="Choose Interviewee: ")

            file_input.change(get_speakers_names, file_input, speaker_choice)

        with gr.Column(1):

            transcript_outputs = [gr.Textbox(label="Transcription")]
            timestamps_outputs = [gr.Textbox(label="Interview Start and End timestamp")]

            speaker_choice.input(
                timestamp_logic, [file_input, speaker_choice], timestamps_outputs
            )

            audio_input.upload(
                transcription_logic,
                [audio_input, file_input, speaker_choice],
                transcript_outputs,
            )
            
            download_button = gr.DownloadButton(label="Download the .txt file (Click me twice for dowload)", value=filepath)
            download_button.click(download_logic, transcript_outputs, download_button)


if __name__ == "__main__":

    demo.launch()
