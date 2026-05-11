#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Income Classification App')
    subparsers = parser.add_subparsers(title='command', dest='command', description='Sub-Command')
    parser.set_defaults(command='serve')

    serve_parser = subparsers.add_parser('serve', help='Serve the App (default)')

    train_parser = subparsers.add_parser('train', help='Train the Model')
    train_parser.add_argument('--config', type=Path, required=True, help='Path to the training configuration file')

    args = parser.parse_args()

    print(f'Args: {args}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
