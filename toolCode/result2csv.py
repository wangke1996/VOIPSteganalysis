import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parameters for data
    parser.add_argument("--result_dir", type=str, default="/data1/wangke/VOIP/results")
    parser.add_argument("--language", type=str, default="ch", help="ch | en")
    parser.add_argument("--method", type=str, default="g729a", help="steganography method")
    parser.add_argument("--duration", nargs='+',type=int, default=[200,400,600,800,1000], help="duration of voice (ms), from 100 to 100000")
    parser.add_argument("--hidden_ratio", nargs='+', type=int, default=[0, 10],help="classes to classify, hidden ratio from 0 to 100")