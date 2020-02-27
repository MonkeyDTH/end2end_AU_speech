'''
Copyright (c) 2017 Hai Pham, Rutgers University
http://www.cs.rutgers.edu/~hxp1/

This code is free to use for academic/research purpose.

'''
import math
import pathlib as plb
import librosa
import csv
import numpy as np

from extract_feature import write_csv, get_fps, extract_one_frame_data

FREQ_DIM = 128
TIME_DIM = 32
NFFT = FREQ_DIM*2

nFrameSize = (TIME_DIM - 3) * FREQ_DIM + NFFT

def extract_one_file(videofile, audiofile):
    print (" --- " + audiofile)
    # get video FPS
    nFrames, fps = get_fps(videofile)
    # load audio
    data, sr = librosa.load(audiofile, sr=44100) # data is np.float32
    # number of audio samples per video frame
    nSamPerFrame = int(math.floor(float(sr) / fps))
    # number of samples per 20ms
    #nSamPerFFTWindow = NFFT #int(math.ceil(float(sr) * 0.02))
    # number of samples per step 8ms
    #nSamPerStep = FREQ_DIM #int(math.floor(float(sr) * 0.008))
    # number of steps per frame
    #nStepsPerFrame = TIME_DIM #int(math.floor(float(nSamPerFrame) / float(nSamPerStep)))
    # real frame size
    #nFrameSize = (nStepsPerFrame - 1) * nSamPerStep + nSamPerFFTWindow
    # initial position in the sound stream
    # initPos negative means we need zero padding at the front.
    curPos = nSamPerFrame - nFrameSize
    dbspecs = []
    for f in range(0,nFrames):
        frameData, nextPos = extract_one_frame_data(data, curPos, nFrameSize, nSamPerFrame)
        curPos = nextPos
        # spectrogram transform
        FD = librosa.core.stft(y=frameData, n_fft=NFFT, hop_length=FREQ_DIM)
        FD, phase = librosa.magphase(FD)
        DB = librosa.core.amplitude_to_db(FD, ref=np.max)
        # scale dB-spectrogram in [0,1]
        DB = np.divide(np.absolute(DB), 80.0)
        # remove the last row
        newDB = DB[0:-1,:]
        # store
        dbspecs.append(newDB.flatten().tolist())
    return dbspecs

video_root = "../Data/RAVDESS/Video"
feat_root = "../Data/RAVDESS/features"
audio_root = "../Data/RAVDESS/Audio"


def process_all():
    video_dir = plb.Path(video_root)
    audio_dir = plb.Path(audio_root)
    feat_dir = plb.Path(feat_root)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for actor in video_dir.iterdir():
        # print(f"[actor] {actor}")
        num = 0
        for video_file in actor.iterdir():
            # print(f"[video_file] {video_file}")
            seq_dir = plb.Path(feat_dir / actor.name / video_file.stem)
            # print(f"[seq_dir] {seq_dir}")
            if not seq_dir.exists():
                seq_dir.mkdir(parents=True)
            audio_path = plb.Path(audio_dir / actor.name / f'{video_file.stem.replace("02", "03", 1)}.wav')
            # print(f'[audio_path] {audio_path}')
            dbspecs = extract_one_file(str(video_file), str(audio_path))
            feature_path = plb.Path(seq_dir / "dbspectrogram.csv")
            # print(f'[feature_path] {feature_path}')
            write_csv(str(feature_path), dbspecs)


if __name__ == "__main__":
    process_all()
