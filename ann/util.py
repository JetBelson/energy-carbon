import os


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log(file, msg='', is_print=True):
    """ Write message msg to file """
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()