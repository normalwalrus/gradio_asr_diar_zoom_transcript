from collections import Counter, defaultdict
import logging
import re

import gradio as gr

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


def read_txt_file(file):
    """
    Read a txt file and return a string
    """
    if file is None:
        return "No file uploaded."
    with open(file.name, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_speakers_names(filepath):
    """
    Get all the names of all speakers from a Zoom template
    """
    text = read_txt_file(filepath)

    pattern = r"^([A-Za-z\s]+): "
    sentence = re.findall(pattern, text, re.MULTILINE)

    speakers = list(set(sentence))
    speaker_checkbox = gr.Radio(speakers)

    return speaker_checkbox

def get_timestamps_for_speaker_timestamps(speaker, file):
    """
    Get the start and end timestamps for a given speaker (from zoom transcript template)

    returns start_timestamp, end_timestamp and all timestamps found
    """
    text = read_txt_file(file)

    pattern = (
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\n"
        + re.escape(speaker)
        + ":"
    )
    matches = re.findall(pattern, text)

    print(f"All matches found: {matches}")

    start_time = matches[0][0]
    end_time = matches[-1][1]

    return start_time, end_time, matches



def get_timestamps_for_speaker(file):
    """
    Get the start and end timestamps for a given speaker (from zoom transcript template)

    returns start_timestamp, end_timestamp and all timestamps found
    """
    text = read_txt_file(file)

    pattern = (
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?):"
    )
    matches = re.findall(pattern, text)

    return matches


def replacement_of_string_in_text(text, old_text, replacement_text):
    """
    Replaces old_text in a string (text) with the new replacement_text using Regex
    """
    pattern = re.compile(rf"{re.escape(old_text)}")
    return pattern.sub(replacement_text, text)


def convert_to_seconds(timestamp):
    """
    Convert a timestamp in the format 'HH:MM:SS.mmm' to seconds.
    """
    hours, minutes, seconds = map(float, timestamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def convert_list_of_timestamps_to_seconds(time_intervals):
    """
    Takes the time intervals of speech and converts it to a list of averages between the 2 time stamps
    """

    average_intervals = [
        [((convert_to_seconds(start) + convert_to_seconds(end)) / 2), speaker]
        for start, end, speaker in time_intervals
    ]

    return average_intervals


def convert_diar_string_to_list(text, offset):
    """
    Converts Diarization to timestamp and speakers. E.g.

    [199.34 - 205.00] [SPEAKER_02] :  Okay, thanks. I'll let the other panelists ask some questions first. I'll get back to you again.\n\n

    TO

    [232.68, 238.34, 'SPEAKER_02']

    Numbers are different because of offset (Should be the start of speech from speaker)

    """

    pattern = r"\[(\d+\.\d+) - (\d+\.\d+)\] \[([A-Z_0-9]+)\]"

    matches = re.findall(pattern, text)
    new_matches = []

    for match in matches:
        match_entry = []

        match_entry.append(float(match[0]))
        match_entry.append(float(match[1]))
        match_entry.append(match[2])

        match_entry[0] += offset
        match_entry[1] += offset

        new_matches.append(match_entry)

    return new_matches


def most_frequent_in_list(List):
    """
    Finds most frequent element in the list
    """
    return max(set(List), key=List.count)


def get_most_frequent_speaker(transcription_time_segments, average_actual_time_sec):
    """
    Given the transcription time segments and the average time from the actual zoom transcript,
    we go through the whole audio to find out which speaker is our guy.
    """

    final_speakers = []

    for average_time in average_actual_time_sec:

        for transcription_time_segment in transcription_time_segments:

            if (
                average_time[0] >= transcription_time_segment[0]
                and average_time[0] <= transcription_time_segment[1]
            ):

                final_speakers.append([transcription_time_segment[2], average_time[1]])

                continue
    
    
    speaker_names = defaultdict(list)
    for speaker, name in final_speakers:
        speaker_names[speaker].append(name)

    # Step 2: Find the most common name for each speaker
    speaker_most_common = {}
    for speaker, names in speaker_names.items():
        most_common_name = Counter(names).most_common(1)[0][0]  # Get the most common name
        speaker_most_common[speaker] = most_common_name

    return speaker_most_common


def download_string_as_txt(string):
    '''
    Not used util function
    '''
    
    output_filepath = f"transcription.txt"

    with open(output_filepath, "w") as text_file:
        text_file.write(string)

    return output_filepath