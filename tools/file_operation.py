import os
import sys

def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    从目录下获取所有文件的路径，同时还能避免重复。
    """
    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path,file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_set = set()
    for ext in file_exts:
        file_path_set = file_path_set | set(get_file_name(os.path.join(base_dir, dir_ext), ext))

    return file_path_set

def read_pickle(path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))

def dump_pickle(data, path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True

def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        sys.stderr.write(e)
        sys.exit(1)
