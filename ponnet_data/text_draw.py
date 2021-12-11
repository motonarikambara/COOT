import subprocess
import json
import ffmpeg


# json_file = "./text_video/translations_3_val.json"

# # text encodeing
# with open(json_file, "r") as f:
#     res = json.load(f)
# print(res)

# for clip_id in res["results"]:
#     file_name = "./samples/_" + str(clip_id) + ".mp4"
#     newfile_name = "./text_video/" + str(clip_id) + ".mp4"
#     pred = ''.join(res["results"][clip_id][0]["sentence"])
#     gt = ''.join(res["results"][clip_id][0]["gt_sentence"][0])
#     text = "Pred: " + pred + "\nGt: " + gt
#     input = ffmpeg.input(file_name)
#     processed = input.video.filter("drawtext",x=100,y=50,text=text,fontfile="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",fontsize=100,fontcolor='#FFFFFF')
#     ffmpeg.output(processed, input.audio, newfile_name).run()
    # print(text)
    # command = 'drawtext=fontfile=fontfile=/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerif.ttf:text={0}:fontsize=16:x=200:y=200'.format(text)
    # ffmpeg = 'ffmpeg -i {0} -filter_complex "{1}" {2}'.format(file_name, command, newfile_name)
    # print(ffmpeg)
    # subprocess.call(ffmpeg, shell=True)