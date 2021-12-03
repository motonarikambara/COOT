import sys
import argparse
import json
import csv


def make_rawdatajson(raw_csv, raw_json):
    json_list = []
    keys = ('clip_id', 'sentence')

    # CSV ファイルの読み込み
    with open(raw_csv, 'r') as f:
        for row in csv.DictReader(f, keys):
            json_list.append(row)

    # JSON ファイルへの書き込み
    with open(raw_json, 'w') as f:
        json.dump(json_list, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--json", type=str)
    args = parser.parse_args()
    make_rawdatajson(args.csv, args.json)


if __name__ == "__main__":
    main()