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

"""Processing data for pretraining."""

import sys
import argparse
import json
import time
import multiprocessing

import torch

try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from transformers import tokenizer
import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # initialize splitter
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

        # initialize tokenizer
        Encoder.tokenizer = tokenizer(self.args.tokenizer)

    def encode(self, text):
        doc_ids = []
        for sent in Encoder.splitter.tokenize(text):
            sent_ids = Encoder.tokenizer.tokenize(sent)
            if len(sent_ids) > 0:
                doc_ids.append(sent_ids)

        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.eod)

        return doc_ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--split_sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep_newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       help='What type of tokenizer to use.')
    group.add_argument('--append_eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--output_prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker processes to launch')
    args = parser.parse_args()
    #args.keep_empty = False

    return args

def main():
    args = get_args()

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)

    fhdl = open(args.input, 'r', encoding='utf-8')
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.initializer)
    enc_docs = pool.imap(encoder.encode, fhdl, 25)

    bin_file = "{}.dat".format(args.output_prefix)
    idx_file = "{}.idx".format(args.output_prefix)
    index = Index()
    build.build(enc_docs, bin_file, idx_file)


if __name__ == '__main__':
    main()
