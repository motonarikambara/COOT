import subprocess
import csv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from s3dg import S3D

# tsv_file = "150samples.tsv"

# ffmpegによる動画の切り出し
# with open(tsv_file, 'r') as f:
#     reader = csv.reader(f, delimiter="\t")
#     i = 0
#     for row in reader:
#         duration = row[4]
#         start = row[2]
#         file_name = str(i) + ".mp4"
#         ffmpeg = "ffmpeg -ss {0} -i headcamera.mp4 -t {1} -c copy {2}".format(start, duration, file_name)
#         subprocess.call(ffmpeg, shell=True)
#         i += 1

# 動画のレート変更
# for i in range(400):
#     file_name = str(i) + ".mp4"
#     newfile_name = "_" + str(i) + ".mp4"
#     ffmpeg = "ffmpeg -i {0} -r 32 {1}".format(file_name, newfile_name)
#     subprocess.call(ffmpeg, shell=True)

# S3Dを用いた動画埋め込み
net = S3D('s3d_dict.npy', 512)
net.load_state_dict(torch.load('s3d_howto100m.pth'))
net = net.eval()
vid_arr = []
for i in tqdm(range(340)):
    tmp_vid = []
    newfile_name = "_" + str(i) + ".mp4"
    cap_file = cv2.VideoCapture(newfile_name)
    while True:
        ret, frame = cap_file.read()
        if not ret:
            break
        frame = cv2.resize(frame, dsize=(224, 224))
        tmp_vid.append(frame)
    video = torch.tensor(tmp_vid, dtype=torch.float64).view(1, 3, -1, 224, 224)
    video_output = net(video)
    vid_arr.append(video_output)

video_feat = np.array(vid_arr)
print(video_feat.shape)
