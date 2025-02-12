import logging

import librosa

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)


def resample_audio_filepath(audio_filepath, desired_sr):

    logging.info("Audio preprocessing starte : To %s SR", desired_sr)

    y, sr = librosa.load(audio_filepath)
    y_desired = librosa.resample(y, orig_sr=sr, target_sr=desired_sr)
    y_mono = librosa.to_mono(y_desired)

    logging.info("Audio preprocessed, Shape : %s", y_mono.shape)

    return y_mono


def resample_audio_array(y, original_sr, desired_sr):

    logging.info("Audio preprocessing starte : %s SR to %s SR", original_sr, desired_sr)

    y_desired = librosa.resample(y, orig_sr=original_sr, target_sr=desired_sr)
    y_mono = librosa.to_mono(y_desired)

    logging.info("Audio preprocessed, Shape : %s", y_mono.shape)

    return y_mono
