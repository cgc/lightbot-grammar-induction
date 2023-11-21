import re
import os
import pathlib
from rrtd_generate import *

code_dir = pathlib.Path(__file__).absolute().parent
exp_dir = code_dir / '../../..' / 'cocosci-lightbot'

def replace_num_conds(fn, num_conds):
    with open(fn) as f:
        lines = f.readlines()

    new_lines = []
    replaced = 0
    for line in lines:
        if line.startswith('num_conds = '):
            line = re.sub('num_conds = \d+\n', f'num_conds = {num_conds}\n', line)
            replaced += 1
        new_lines.append(line)
    assert replaced == 1, f'Expected one replacement, but instead had {replaced}'

    with open(fn, 'w') as f:
        f.write(''.join(new_lines))

def set_current_config_json(config_file):
    dst = exp_dir / 'static/lightbot/json/configuration.json'
    if dst.exists():
        os.unlink(dst)
    os.symlink(config_file.relative_to(dst.parent), dst)
