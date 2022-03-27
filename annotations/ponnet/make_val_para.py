import json

with open("captioning_train.json", "r") as f:
    cap_data = json.load(f)
val_dict = {}
for i in cap_data:
    val_dict[i["clip_id"]] = i["sentence"]

with open('captioning_train_para.json', 'w') as f:
    json.dump(val_dict, f)