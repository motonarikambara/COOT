import subprocess
import csv
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os

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
# for i in range(340):
#     file_name = str(i) + ".mp4"
#     newfile_name = "_" + str(i) + ".mp4"
#     ffmpeg = "ffmpeg -i {0} -r 4 {1}".format(file_name, newfile_name)
#     subprocess.call(ffmpeg, shell=True)

# S3Dを用いた動画埋め込み
net = S3D('s3d_dict.npy', 512).cuda()
net.load_state_dict(torch.load('s3d_howto100m.pth'))
net = net.eval()
for i in tqdm(range(340)):
    tmp_vid = []
    newfile_name = "_" + str(i) + ".mp4"
    cap_file = cv2.VideoCapture(newfile_name)
    video = torch.zeros((1, 3, 32, 224, 224))
    for j in range(32):
        ret, frame = cap_file.read()
        if not ret:
            break
        frame = cv2.resize(frame, dsize=(224, 224))
        frame = torch.tensor(frame, dtype=torch.float32).view(3, 224, 224)
        video[0, :, j, :, :] = frame
    cap_file.release()
    video = video.cuda()
    video_output = net(video)
    emb_feat = video_output["video_embedding"].to('cpu').detach().numpy().copy()
    del video
    del video_output
    torch.cuda.empty_cache()
    file_name = os.path.join("emb_feats", str(i))
    np.save(file_name, emb_feat)
