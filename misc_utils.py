import os


def get_unmatched_ids(dirs, split_char="-"):
    '''
    returns a list of all ids that do not exist in all given dirs
    '''
    files = [os.listdir(dir) for dir in dirs]
    file_ids = [[file.split(split_char)[0] for file in file_list]for file_list
                in files]
    sets = [set(file_id) for file_id in file_ids]
    unmatched = [id_set - set().union(*(sets[:i] + sets[i+1:])) for i, id_set
                 in enumerate(sets)]
    unmatched = set().union(*unmatched)
    return unmatched


def get_matched_ids(dirs, split_char="-"):
    '''
    returns a sorted set of all ids that exist in all given dirs
    '''
    files = [os.listdir(dir) for dir in dirs]
    file_ids = [[file.split(split_char)[0] for file in file_list] for
                file_list in files]
    sets = [set(file_id) for file_id in file_ids]
    matched = set.intersection(*sets)
    return sorted(matched)
