import os
import pickle
import hashlib
import os.path as osp

path_to_train = 'data/cifar100/cifar-100-python/train'

cifar100_original_train = pickle.load(open(path_to_train, 'rb'),
                                      encoding='latin1')
cifar100_new_train = {}
cifar100_new_val = {}


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


for k in cifar100_original_train:
    if len(cifar100_original_train[k]) == 50000:
        cifar100_new_train[k] = cifar100_original_train[k][:40000]
        cifar100_new_val[k] = cifar100_original_train[k][40000:]
    else:
        cifar100_new_train[k] = cifar100_original_train[k]
        cifar100_new_val[k] = cifar100_original_train[k]

os.makedirs(osp.join('data', 'cifar100-new-split',), exist_ok=True)
new_train_path = osp.join('data', 'cifar100-new-split', 'cifar-100-python',
                          'train')
with open(new_train_path, 'wb') as f:
    pickle.dump(cifar100_new_train, f)
print(f"train set has md5: {calculate_md5(new_train_path)}")
new_val_path = osp.join('data', 'cifar100-new-split', 'val')
with open(new_val_path, 'wb') as f:
    pickle.dump(cifar100_new_val, f)
print(f"validation set has md5: {calculate_md5(new_val_path)}")

path_to_test = 'data/cifar100/cifar-100-python/test'
path_to_new_test = osp.join('data', 'cifar100-new-split', 'cifar-100-python',
                            'test')
with open(path_to_new_test, 'wb') as f:
    with open(path_to_test, 'rb') as g:
        pickle.dump(pickle.load(g, encoding='latin1'), f)
print(f"test set has md5: {calculate_md5(path_to_new_test)}")

path_to_meta = 'data/cifar100/cifar-100-python/meta'
path_to_new_meta = osp.join('data', 'cifar100-new-split', 'meta')
with open(path_to_new_meta, 'wb') as f:
    with open(path_to_meta, 'rb') as g:
        pickle.dump(pickle.load(g, encoding='latin1'), f)
print(f"meta has md5: {calculate_md5(path_to_new_meta)}")