"""
エラー分析用スクリプト
結果のファイルからタイムスタンプ，生成文，正解文を取り出して
出力用ファイルにクリップごとに並べて出力
結果ディレクトリ：./experiments/caption/default/yc2_100m_coot_clip_mart_run(日時)
                /caption/
"""

import sys
import json


def compare_res(res_path, out_path):
    """
    Args:
        res_path(string): 結果のファイルパス
        out_path(string): エラー分析のための出力ファイルパス
    """
    with open(res_path, "r") as f:
        res = json.load(f)
    res = res["results"]
    with open(out_path, "w") as f:
        f.write("timestamp: gen_sent  gt_sent\n")
        for vid in res:
            video = res[vid]
            for clip in video:
                time = clip["timestamp"]
                gen_sent = clip["sentence"]
                gt_sent = clip["gt_sentence"]
                compare = "{}: {}  {}\n".format(time, gen_sent, gt_sent)
                f.write(compare)


def main(paths):
    res_path = paths[1]
    out_path = paths[2]
    compare_res(res_path, out_path)


if __name__ == '__main__':
    args = sys.argv
    paths = []
    for path in args:
        paths.append(path.rstrip('\r\n'))
    main(paths)
