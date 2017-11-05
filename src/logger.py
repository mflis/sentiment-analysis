import subprocess

import src


def print_source(file):
    log_file = '../sources_log/{}'.format(src.CURRENT_TIME)
    git_info = subprocess.check_output(['git', 'log', '-1', '--oneline'])

    with open(log_file, mode='w') as log:
        with open(file) as f:
            log.write(f.read())
            log.write('# {}'.format(git_info))
