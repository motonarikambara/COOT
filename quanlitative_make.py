import json
import csv

target_file = open("./captions/first_lstm.json", "r")
target_text = json.load(target_file)

with open("./captions/first_lstm.csv", "w") as f:
    writer = csv.writer(f)
    header = []
    header.append("ID")
    header.append("timestamp")
    header.append("generated")
    header.append("GT")
    writer.writerow(header)
    for i in target_text["results"].keys():
        for j in range(len(target_text["results"][i])):
                row = []
                row.append(i)
                row.append(target_text["results"][i][j]["timestamp"])
                row.append(target_text["results"][i][j]["sentence"])
                row.append(target_text["results"][i][j]["gt_sentence"])
                writer.writerow(row)