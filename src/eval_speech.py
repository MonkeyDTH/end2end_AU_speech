'''
Copyright (c) 2018 Hai Pham, Rutgers University
http://www.cs.rutgers.edu/~hxp1/

This code is free to use for academic/research purpose.

'''

import numpy as np
import cntk as C
import cv2
from scipy.signal import medfilt

import ShapeUtils2 as SU
from SysUtils import make_dir, get_items, get_current_time_string


def load_image(path):
    img = cv2.imread(path, 1)
    if img.shape[0] != 100 or img.shape[1] != 100:
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    return img

def load_image_stack(paths):
    imgs = [load_image(path) for path in paths]
    return np.stack(imgs)


def load_exp_sequence(path, use_medfilt=False, ksize=3):
    exp = np.load(path).astype(np.float32)
    if use_medfilt:
        exp = medfilt(exp, kernel_size=(ksize,1)).astype(np.float32)
    return exp


# estimate audio sequence -> exp
def is_recurrent(model):
    names = ["RNN", "rnn", "LSTM", "LSTM1", "GRU", "gru", "fwd_rnn", "bwd_rnn"]
    isrnn = False
    for name in names:
        if model.find_by_name(name) is not None:
            isrnn = True
    return isrnn


def estimate_one_audio_seq(model, audio_seq, small_mem=False):
    if isinstance(model, str):
        model = C.load_model(model)
    # set up 2 cases: if the model is recurrent or static
    if is_recurrent(model):
        n = audio_seq.shape[0]
        NNN = 125
        if n > NNN and small_mem:
            nseqs = n//NNN + 1
            indices = []
            for i in range(nseqs-1):
                indices.append(NNN*i + NNN)
            input_seqs = np.vsplit(audio_seq, indices)
            outputs = []
            for seq in input_seqs:
                output = model.eval({model.arguments[0]:[seq]})[0]
                outputs.append(output)
            output = np.concatenate(outputs)
        else:
            output = model.eval({model.arguments[0]:[audio_seq]})[0]
    else:
        output = model.eval({model.arguments[0]: audio_seq})
    return output


#----------------------- feed sequence -------------------------
def visualize_one_audio_seq(model, video_frame_list, audio_csv_file, exp_npy_file, visualizer, save_dir):
    if isinstance(model, str):
        model = C.load_model(model)
    # evaluate model with given audio data
    audio = np.loadtxt(audio_csv_file, dtype=np.float32, delimiter=",")
    audio_seq = np.reshape(audio, (audio.shape[0], 1, 128, 32))
    e_fake = estimate_one_audio_seq(model, audio_seq)
    if e_fake.shape[1] != 46:
        if e_fake.shape[1] == 49:
            e_fake = e_fake[:, 3:]
        else:
            raise ValueError("unsupported output of audio model")
    # load true labels with optional median filter to smooth it (not used in training)
    e_real = load_exp_sequence(exp_npy_file, use_medfilt=True)

    if e_real.shape[0] > e_fake.shape[0]:
        length = e_fake.shape[0]
        e_real = e_real[:length]

    if e_real.shape[0] < e_fake.shape[0]:
        length = e_real.shape[0]
        e_fake = e_fake[:length]

    if e_real.shape[0] != e_fake.shape[0]:
        raise ValueError("number of true labels and number of outputs do not match")
    
    # create directory to store output frames
    if video_frame_list:
        video = load_image_stack(video_frame_list)
        if video.shape[0] != e_real.shape[0]:
            print("number of frames and number of labels do not match. Not using video")
            video = None
    else:
        video = None
    # make folder to store generated frames
    make_dir(save_dir)

    n = e_real.shape[0]
    for i in range(n):
        if video is not None:
            img = video[i, :, :, :]
        else:
            img = None  # not include input video in the output
        ef = e_fake[i, :]
        er = e_real[i, :]
        ret = visualizer.visualize(img, er, ef)
        # draw plot
        plot = SU.draw_error_bar_plot(er, ef, (ret.shape[1],200))
        ret = np.concatenate([ret, plot], axis=0)
        save_path = save_dir + "/result{:06d}.jpg".format(i)
        cv2.imwrite(save_path, ret)
        # can call cv2.imshow() here


#----------------------------------------------------------------------------------------
def test_one_seq(visualizer):
    """ Test one sequence """
    ''' Load model '''
    # model_file = "../Model/model_audio2exp_2020-02-25-13-02/model_audio2exp_2020-02-25-13-02.dnn"
    model_file = "../Model/model_audio2exp_2020-02-25-15-07/model_audio2exp_2020-02-25-15-07.dnn"  # GRU
    model = C.load_model(model_file)

    ''' Set input and output dir '''
    current_time = get_current_time_string()
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
                    print(f"[Start] sample: {sample}")
                    save_dir = f"../Test_output/GRU_{sample}"
                    # video directory holding separate frames of the video. Each image should be square.
                    video_dir = f"../Data/RAVDESS/Video_Frame/Actor_01/{sample}"
                    # spectrogram sequence is stored in a .csv file
                    audio_file = f"../Data/RAVDESS/features/Actor_01/{sample}/dbspectrogram.csv"
                    # AU labels are stored in an .npy file
                    exp_file = f"../Data/ExpLabels/RAVDESS/Actor_01/{sample}.npy"

                    video_list = get_items(video_dir, "full")  # set to None if video_dir does not exist
                    visualize_one_audio_seq(model, video_list, audio_file, exp_file, visualizer, save_dir)

    # filename = "01-01-01-01-01-01-01"
    # # directory to store output video. It will be created if it doesn't exist
    # save_dir = f"../Test_output/GRU_{filename}"
    # # video directory holding separate frames of the video. Each image should be square.
    # video_dir = f"../Data/RAVDESS/Video_Frame/Actor_01/{filename}"
    # # spectrogram sequence is stored in a .csv file
    # audio_file = f"../Data/RAVDESS/features/Actor_01/{filename}/dbspectrogram.csv"
    # # AU labels are stored in an .npy file
    # exp_file = f"../Data/ExpLabels/RAVDESS/Actor_01/{filename}.npy"

    # # single test
    # save_dir = "../Data/test_output_single"
    # # model_file = "../Model/model_audio2exp_2020-02-25-13-02/model_audio2exp_2020-02-25-13-02.dnn"
    # model_file = "../Model/model_audio2exp_2020-02-25-15-07/model_audio2exp_2020-02-25-15-07.dnn"
    # video_dir = "../Data/RAVDESS/Video_Frame/Actor_01/01-01-01-01-01-01-01"
    # audio_file = "../Data/RAVDESS/features/Actor_01/01-01-01-01-01-01-01/dbspectrogram.csv"
    # exp_file = "../Data/ExpLabels/RAVDESS/Actor_01/01-01-01-01-01-01-01.npy"

    # video_list = get_items(video_dir, "full")  # set to None if video_dir does not exist
    # model = C.load_model(model_file)
    #
    # visualize_one_audio_seq(model, video_list, audio_file, exp_file, visualizer, save_dir)

    print("Eval finished!")


#----------------------------------------------------------------------------------
if __name__ == "__main__":
    visualizer = SU.Visualizer()
    test_one_seq(visualizer)
