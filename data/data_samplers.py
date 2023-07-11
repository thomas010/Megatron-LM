# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataloaders."""


import torch
import random
from megatron import get_args
from megatron import mpu

import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset
import torch.distributed as dist

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)



class DistributedBatchSampler(Sampler[T_co]):
    r"""batch sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int,
                 num_steps: int, micro_batch_size: int, seq_length: int, seed: int = 0) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_steps = num_steps
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[T_co]:
        steps = 0
        epoch = 0
        batch = []
        cur_sample = []
        cur_length = 0
        while steps < self.num_steps:
            epoch = epoch + 1
            self.set_epoch(epoch)

            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = indices[self.rank::self.num_replicas]

            for idx in indices:
                data = self.dataset[idx]
                size = self.dataset.sizes[idx]
                pointer = 0
                while pointer < size:
                    if self.seq_length-cur_length <= size-pointer:
                        sample.extend(data[pointer:pointer+self.seq_length-cur_length])
                        pointer += self.seq_length-cur_length
                        batch.append(cur_sample)
                        cur_sample = []
                        cur_length = 0
                        if len(batch) == self.micro_batch_size:
                            yield batch
                            steps += 1
                            batch = []
                    else:
                        sample.extend(data[pointer:])
                        cur_length += size-pointer

    def __len__(self) -> int:
        return self.num_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def multi_data_loader():
    torch.multinomial(weights, 10000)