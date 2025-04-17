import lmdb
import os
from tqdm import tqdm
import argparse
from basicsr.utils import scandir


base_dir =r"/media/dy/ext4/yuandi/VOS"

# namelist = list(scandir(base_dir, suffix=('label','ini'), recursive=True))
namelist = list(scandir(base_dir, recursive=True))
print('number:', len(namelist))

# if base_dir.endswith("/"):
#     lmdb_fname = base_dir[:-1] + '_lmdb'
# else:
#     lmdb_fname = base_dir + '_lmdb'

lmdb_fname = '/media/dy/sda2/dataset/Youtubevos_lmdb'
env = lmdb.open(lmdb_fname, map_size=1024 ** 4)
txn = env.begin(write=True)

for i, t in enumerate(tqdm(namelist)):
    if i % 100000 == 0:
        txn.commit()
        txn = env.begin(write=True)
    with open(os.path.join(base_dir, t), 'rb') as fin:
        txn.put(key=t.encode(), value=fin.read())

txn.commit()
env.close()
