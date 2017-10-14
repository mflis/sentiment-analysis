import argparse


def maybe_cut_args(*arrays):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut', action='store')
    args = parser.parse_args()
    return [x[:args.cut, :] for x in arrays]
