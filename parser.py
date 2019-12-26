from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='My parser')

    args = parser.parse_args()

    return args
