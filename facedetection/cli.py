import argparse

from luigi import build

from .task import Aggregation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", help="input directory", type=str, default="data/asian")
    parser.add_argument('dst_path', help="output directory", type=str, default="data/asianresult")
    args = parser.parse_args()
    raw_path = args.raw_path
    dst_path = args.dst_path

    build([
        Aggregation(raw_path, dst_path)
    ], local_scheduler=True)