import argparse
import datetime
import glob
import math
import ntpath
import os
import shutil

import numpy as np
import pandas
from mne.io import read_raw_edf
import mne


W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5


stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}#建立一个字典


class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}


ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}#将S3和S4合并为S3

EPOCH_SEC_SIZE = 30
def save_npz(sub_x, sub_y, sub_cnt, sfrep, ref_chan, save_dir, serious, filename):
    """
    保存npz文件
    :param sub_x: 被试数据list: [sub_parts * (epochs, channels, samples)]
    :param sub_x: 被试标签list: [sub_parts * (epochs,)]
    :param sub_cnt: 被试睡眠事件统计array: (5,)
    :param sfrep: 采样率int     :param ref_chan: 参考后的电极list[9] :param save_dir: 保存地址
    :param filename: 保存的文件名:return:sub_x(epochs', channels, samples),sub_y(epochs',channels,samples)
    """
    # np.vstack按垂直方向堆叠数组，相当于再添一个维度；hstack就不添维度
    sub_x = np.vstack(sub_x)  # list:sub_parts*(epochs, channel, samples),-->(1214,9,7680)
    sub_y = np.hstack(sub_y)  # list:sub_parts*(epochs,)
    save_dict = {
        "x": sub_x,
        "y": sub_y,
        "fs": sfrep,
        "ch_label": ref_chan,
        "stages_cnt": sub_cnt
    }
    output_dir=save_dir
    #output_dir = os.path.join(save_dir, serious)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez_compressed(os.path.join(output_dir, filename), **save_dict)
    print("SAVE:{}.npz successfully!".format(filename))

#将EDF+文件处理成npz文件
def main():
    """This function convert EDF+ files to npz file.
    """
    # Preparing args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/home/denglongyan/papercode/data/MASS/SS3",
                        help="File path to the edf file that contain sleeping info.")
    parser.add_argument("--output_dir", "-o", type=str, default="/home/denglongyan/papercode/data/ss3_npz",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", '-s', type=list, default=["EEG F4-LER"],
                        help="Choose the channels for training.")#选择三个通道，跟edfbrowser中的通道一致
    # default=["EEG F4-LER", "EEG C4-LER", "ECG ECGI"],
    args = parser.parse_args()
    """
    国际标准导联：F4-M1, C4-M1, O2-M1
    """
    # output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channels
    select_ch: list = args.select_ch#声明一个变量select_ch,类型为列表

    # Read raw and annotation EDF files.
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Base.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i, file in enumerate(psg_fnames):#使用enumerate获取文件在列表中的索引(0,1,2,...)和文件名
        #print(file.split('-', 2)[2].split(' ', 1)[0])
        raw = read_raw_edf(file, preload=True, stim_channel=None)#使用MNE-Python库中的read_raw_edf函数读取原始的edf文件,preload代表将数据加载到内存,不指定通道
        sampling_rate = raw.info['sfreq']#原始的EDF+文件含有很多信息(raw.info)，这里只获取它的频率信息 MASS 256
        raw = raw.notch_filter(50)
        # 带通滤波:心电信号滤波范围(0.3,70),眼电信号滤波范围(0.3,35)，脑电信号滤波范围(0.3,35)，肌电信号滤波范围(10,100)
        raw.filter(0.1, 70., fir_design='firwin')  ##??是否要进行 raw.plot_psd()
        # 降采样，不降采样有个512有个256不一致
        raw=raw.resample(100)
        sfrep = 100
        # 进行通道选择，并将七转换为dataframe类型
        raw = raw.pick_channels(select_ch)
        raw_ch_arr = np.array(raw.to_data_frame()[select_ch].T)#[samples, channels]
        # if not isinstance(raw_ch_df, pandas.DataFrame):
        #     raw_ch_df = raw_ch_df.to_frame()
        # raw_ch_df.set_index(np.arange(len(raw_ch_df)))#给采样后的数据帧设置索引

        # 加载标签数据
        annotations = mne.read_annotations(ann_fnames[i])
        # 获取标签的时间戳和描述
        """
        onset：注释的开始时间，通常是指从记录开始到注释开始的时间（秒为单位）。这表示了某个特定事件或睡眠阶段的开始时间点。
        duration：注释的持续时间，以秒为单位。这表示了某个事件或睡眠阶段持续的时间长度。
        description：注释的描述，通常是一个字符串，描述了该时间段的事件类型或睡眠阶段（例如，“Sleep stage W”表示觉醒阶段，"Movement time"表示运动时间等）。
        """
        onset = annotations.onset
        duration = annotations.duration
        description = annotations.description

        # 加载信号数据
        #raw = mne.io.read_raw_edf(ann_fnames[i])
        # 提取标签对应的信号数据
        data = []
        labels = []
        data_with_labels = []
        # 遍历每个标签，找到对应的信号片段并存储
        epoch_cnt = [0,0,0,0,0]
        overstrip_cnt = 0
        for j in range(len(onset)):
            if description[j] == 'Sleep stage ?' or description[j] == 'Movement time':
                overstrip_cnt += 1
                continue
            # start_time = onset[j]
            # end_time = onset[j] + duration[j]
            # signal_segment = raw.get_data(start=start_time.astype(int), stop=end_time.astype(int), return_times=False)

            start_idx = raw.time_as_index(onset[j])
            end_idx = raw.time_as_index(onset[j] + duration[j])
            signal_segment = raw_ch_arr[:, start_idx[0]:end_idx[0]]
            signal_segment = signal_segment[np.newaxis, ...]
            # 统计每个睡眠阶段出现次数，一个duration对应一个标签，所以是+1
            epoch_cnt[ann2label[description[j]]] += 1
            label = ann2label[description[j]] #转换成int类型标签
            data.append(signal_segment)
            labels.append(label)
            data_with_labels.append((signal_segment, label))
        # 将列表转换为numpy数组
        data_with_labels = np.array(data_with_labels, dtype=object)
        # 保存numpy数组为文件
        #np.save('data_with_labels.npy', data_with_labels)
        save_npz(sub_x=data, sub_y=labels, sub_cnt=epoch_cnt, sfrep=100, ref_chan=args.select_ch, save_dir=args.output_dir, serious=' ',filename=file.split('-', 2)[2].split(' ', 1)[0])
if __name__ == '__main__':
    main()