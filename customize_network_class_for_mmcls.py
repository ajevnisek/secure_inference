"""This script takes a class describing a pytorch network and converts it into mmcls backbone."""
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Change the class in filepath class declaration and put it in mmcls's dir.")
    parser.add_argument("filepath", help="Path to the class declaration file.")
    parser.add_argument("mmcls_backbone_dir", help="Path to the mmcls directory.", default="mmpretrain/mmcls/models/backbones/")
    return parser.parse_args()

def create_class_in_backbones(filepath, dirname):
    with open(filepath, 'r') as f:
        data = f.read()
    splited = data.split('class ')
    new_data = splited[0] + '\n' + 'from ..builder import BACKBONES' + '\n\n' + '@BACKBONES.register_module()\nclass ' + '\n'.join(splited[1: ])
    new_filename = os.path.join(dirname, os.path.basename(filepath))
    with open(new_filename, 'w') as f:
        f.write(new_data)


def add_to_init(filepath, dirname, ):
    with open(filepath, 'r') as f:
        data = f.read()
    splited = data.split('class ')
    classname = splited[1].split('(')[0]
    original_init_path = os.path.join(dirname, '__init__.py')
    with open(original_init_path, 'r') as f:
        init_data = f.read()
    lines = init_data.splitlines()
    idx = [pos for pos, l in enumerate(lines) if l.startswith('from')][-1]
    prev_from_lines = lines[:idx + 1]
    new_from_line = f'from .{os.path.basename(filepath)[:-3]} import {classname}'
    list_all_content = lines[idx + 1:-1]
    new_init_data = '\n'.join(prev_from_lines + [new_from_line] + list_all_content + [f'    "{classname}", \n]'])
    with open(original_init_path, 'w') as f:
        f.write(new_init_data)


if __name__ == '__main__':
    args = parse_arguments()
    create_class_in_backbones(args.filepath, args.mmcls_backbone_dir)
    add_to_init(args.filepath, args.mmcls_backbone_dir)
