# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

# Some of the fixes/improvements are adopted from
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/data/indexed_dataset.py

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch
from megatron import print_rank_0


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

class Index(object):
    def __init__(self):
        self._HDR_MAGIC = b'MMIDIDX\x00\x00'

        self.sizes = []
        self.split = []
        self.offset = []
        self.num_sent = 0
        self.num_doc = 0

    def convert_size_to_offset(self, sizes):
        offsets = np.array(sizes, dtype=np.int64)
        np.cumsum(offsets, axis=0, out=offsets)
        offsets[1:] = offsets[:-1]
        offsets[0] = 0
        return offsets

    def save(self, idx_file):
        with open(idx_file, 'wb') as stream:
            stream.write(self._HDR_MAGIC)
            stream.write(struct.pack('<Q', self.num_sent))
            stream.write(struct.pack('<Q', self.num_doc))
            stream.write(np.array(self.sizes, dtype=np.int32).tobytes(order='C'))
            stream.write(np.array(self.split, dtype=np.int32).tobytes(order='C'))

    def read(self, idx_file, warmup=True):
        with open(idx_file, 'wb') as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, 'Index file does not match expected format.'
            self.num_sent = struct.unpack('<Q', stream.read(8))[0]
            self.num_doc = struct.unpack('<Q', stream.read(8))[0]
            offset = stream.tell()

        if warmup:
            _warmup_mmap_file(idx_file)

        self.buffer_mmap = np.memmap(idx_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)
        self.sizes = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_sent, offset=offset)
        self.split = np.frombuffer(self.buffer, dtype=np.int32, count=self.num_doc,  offset=offset + self.sizes.nbytes)
        self.offset = self.convert_size_to_offset(self.sizes)

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self.offset[i], self.sizes[i]

    def __len__(self):
        return self.num_sent

    def build(self, enc_docs, bin_file, idx_file):
        with open(bin_file, 'wb') as stream:
            sum_bytes = 0
            for i, (doc_ids, num_bytes) in enumerate(enc_docs):
                sum_bytes += num_bytes
                if len(doc_ids) == 0:
                    continue

                self.split.append(len(self.sizes))
                self.num_doc += 1

                for sent_ids in doc_ids:
                    np_array = np.array(sent_ids, dtype=np.int32)
                    stream.write(np_array.tobytes(order='C'))
                    self.sizes.append(np_array.size)
                    self.num_sent += 1

        self.save(idx_file)
        print("total encoded: %.2f Mb text | %d sents | %d docs " % (sum_bytes / 1024 / 1024, self.num_sent, self.num_doc))


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, bin_file, idx_file, warmup=True):
        super().__init__()
        #
        self.index = Index()
        self.index.read(idx_file, warmup=warmup)

        if warmup:
            _warmup_mmap_file(bin_file)
        self.buffer_mmap = np.memmap(bin_file, mode='r', order='C')
        self.buffer = memoryview(self.buffer_mmap)

    def __del__(self):
        self.buffer_mmap._mmap.close()
        del self.buffer_mmap
        del self.index

    def __len__(self):
        return len(self.index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        offset, size = self.index[idx]
        data = np.frombuffer(self.buffer, dtype=np.int32, count=size, offset=offset)
        return data

