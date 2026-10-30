"""Streaming entry point: revised TSHM by default; legacy remains selectable."""
import argparse
import importlib
import sys


def select_method(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method', choices=('revised', 'legacy'), default='revised')
    args, remaining = parser.parse_known_args(argv)
    return args.method, remaining


def main():
    method, remaining = select_method(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    module = importlib.import_module('streaming_infer_' + method)
    module.main()


if __name__ == '__main__':
    main()
