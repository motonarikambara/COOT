import json


def compare_res(res_path, out_path):
    with(res_path, "r") as f:
        res = json.load(f)
    res = res["results"]
    for vid in 