# The original part of the fairseq: Copyright (c) Facebook, Inc. and its affiliates.
# The modified and additional parts:
# Copyright (c) 2019 National Institute of Information and Communications Technology.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary

class DictionaryWithBert(Dictionary):
    """
    Dictionary for the BERT tokenizer.
    This class does not append the EOS tokens at the end of sentences.
    """
    def __init__(self, dic):
        super().__init__()
        for attr in ['unk_word', 'pad_word', 'eos_word',
                     'symbols', 'count', 'indices',
                     'bos_index', 'pad_index', 'eos_index', 'unk_index',
                     'nspecial', 'indices', ]:
            setattr(self, attr, getattr(dic, attr))

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=False, reverse_order=False):
        append_eos = False
        ids = super().encode_line(
            line, line_tokenizer, add_if_not_exist,
            consumer, append_eos, reverse_order)
        return ids