import json

with open("captioning_test.json", "r") as f:
    cap_data = json.load(f)
val_dict = {}
for i in cap_data:
    val_dict[i["clip_id"]] = i["sentence"]

with open('captioning_test_para.json', 'w') as f:
    json.dump(val_dict, f)
