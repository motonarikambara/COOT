import subprocess
import json
import ffmpeg


json_file = "./translations_6_test.json"

# # text encodeing
with open(json_file, "r") as f:
    res = json.load(f)
# print(res)

for clip_id in res["results"]:
    file_name = "./samples/_" + str(clip_id) + ".mp4"
    newfile_name = "./text_video/" + str(clip_id) + ".mp4"
    pred = ''.join(res["results"][clip_id][0]["sentence"])
    gt = ''.join(res["results"][clip_id][0]["gt_sentence"][0])
    text = "Pred: " + pred + "\nGt: " + gt
    input = ffmpeg.input(file_name)
    processed = input.video.filter("drawtext",x=100,y=50,text=text,fontfile="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",fontsize=20,fontcolor='#FFFFFF')
    ffmpeg.output(processed, input.audio, newfile_name).run()