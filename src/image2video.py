#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/02/28 11:38
# @Author  : TH
# @Site    : 
# @File    : image2video.py
# @Software: PyCharm
"""
Image -> Video & Video -> Image
"""
import os
import pathlib as plb


def exec_cmd(cmd):
    """ 执行Shell指令 """
    r = os.popen(cmd)
    text = r.read().strip()
    r.close()
    return text


def video2image_crop(video_fname, frame_dir):
    """
    Cut video to images & crop to square
    :param video_fname: Video filename
    :param frame_dir: [Out] Image directory
    """
    out_imgs = frame_dir / "img_%05d.png"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = "ffmpeg"
    cmd += f" -i {video_fname}"
    cmd += f" -vf crop=720:720"  # crop to square
    cmd += f" {out_imgs}"
    exec_cmd(cmd)


def video2audio(video_fname, audio_fname):
    """
    split audio from video
    :param video_fname: Video filename
    :param audio_fname: [Out] Audio filename
    """
    audio_fname.parent.mkdir(parents=True, exist_ok=True)
    cmd = "ffmpeg"
    cmd += f" -i {video_fname}"
    cmd += f" {audio_fname}"
    exec_cmd(cmd)


def image2video(input_img, out_video_noaudio_fname):
    """
    generate video from images
    :param input_img: input image format
    :param out_video_noaudio_fname: output video
    """
    cmd = "ffmpeg -f image2"
    cmd += f" -i {input_img}"
    cmd += " -vcodec libx264"
    cmd += " -r 29.97"
    cmd += f" {out_video_noaudio_fname}"
    exec_cmd(cmd)


def add_audio_tovideo(audio_file, out_video_noaudio_fname, out_video_fname):
    """ Add Audio """
    cmd = "ffmpeg"
    cmd += f" -i {audio_file}"
    cmd += f" -i {out_video_noaudio_fname}"
    cmd += f" {out_video_fname}"
    exec_cmd(cmd)


def main():
    """ Main function: image, video and audio operation """
    modality_list = ["01"]
    vocal_channel_list = ["01"]
    emotion_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    emotion_intensity_list = ["01", "02"]
    statement_list = ["01", "02"]
    repetition_list = ["01", "02"]
    actor_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                  "21", "22", "23", "24"]

    """ Video2image & crop & split audio"""
    # video_dir = plb.Path("../Data/RAVDESS/Video/Actor_01/")
    # frame_dir_parent = plb.Path("../Data/RAVDESS/Video_Frame/Actor_01/")
    # audio_dir = plb.Path("../Data/RAVDESS/Audio/Actor_01/")
    # for emotion in emotion_list:
    #     for emotion_intensity in emotion_intensity_list:
    #         for statement in statement_list:
    #             for repetition in repetition_list:
    #                 if emotion == "01" and emotion_intensity == "02":
    #                     continue
    #                 sample = f"01-01-{emotion}-{emotion_intensity}-{statement}-{repetition}-01"
    #                 video_fname = video_dir / (sample + ".mp4")
    #                 frame_dir = frame_dir_parent / sample
    #                 audio_fname = audio_dir / (sample + ".wav")
    #                 video2image_crop(video_fname, frame_dir)
    #                 video2audio(video_fname, audio_fname)

    # # single file
    # video_fname = "../Data/RAVDESS/Video/Actor_01/01-01-01-01-01-01-01.mp4"
    # frame_dir = plb.Path("../Data/RAVDESS/Video_Frame/Actor_01/01-01-01-01-01-01-01")
    # audio_fname = plb.Path("../Data/RAVDESS/Video/Actor_01/Audio/01-01-01-01-01-01-01.wav")
    # video2image_crop(video_fname, frame_dir)
    # video2audio(video_fname, audio_fname)

    """ Image2video """
    modality_list = ["01"]
    vocal_channel_list = ["01"]
    emotion_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    emotion_intensity_list = ["01", "02"]
    statement_list = ["01", "02"]
    repetition_list = ["01", "02"]
    actor_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                  "21", "22", "23", "24"]

    for emotion in emotion_list:
        for emotion_intensity in emotion_intensity_list:
            for statement in statement_list:
                for repetition in repetition_list:
                    if emotion == "01" and emotion_intensity == "02":
                        continue
                    sample = f"01-01-{emotion}-{emotion_intensity}-{statement}-{repetition}-01"
                    # sample = "01-01-01-01-01-01-01"
                    audio_file = f"../Data/RAVDESS/Audio/Actor_01/{sample}.wav"
                    frame_dir = plb.Path(f"../Test_output/GRU_{sample}")
                    input_imgs = frame_dir / "result%06d.jpg"
                    print(input_imgs)
                    out_video_noaudio_fname = str(frame_dir / "output_noaudio.mp4")
                    out_video_fname = str(frame_dir / "output.mp4")
                    print("Start synthesis")
                    image2video(input_imgs, out_video_noaudio_fname)
                    add_audio_tovideo(audio_file, out_video_noaudio_fname, out_video_fname)

    print("Finished!")


if __name__ == "__main__":
    main()
