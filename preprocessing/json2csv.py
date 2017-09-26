import json, os


def convert_dir_list(dir):
    for dname in os.listdir(dir):
        path = os.path.join(dir, dname)
        if (os.path.isdir(path)):
            save_iterable(Converter(path), os.path.join(path, 'sentences.tok'))


def get_array_from_row(row):
    array_row = list()
    for block in row.get('lstBlock'):
        for token in block.get('lstToken'):
            array_row.append(token.get('word'))

    return array_row


def save_iterable(it, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in it:
            f.write(str(line)+'\n')


class Converter(object):
    def __init__(self, parentdir):
        self.parentdir = parentdir


    # This method could raise an error if no sentences were found.
    def __iter__(self):
        for dirpath, dnames, fnames in os.walk(self.parentdir):
            for fname in fnames:
                if fname.endswith("sentences.json"):
                    with open(os.path.join(dirpath, fname), encoding='utf-8') as data_file:
                        for row in json.load(data_file):
                            yield get_array_from_row(row)


dir = 'C:/datasets/es'

convert_dir_list(dir)