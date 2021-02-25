# The original part of the fairseq: Copyright (c) Facebook, Inc. and its affiliates.
# The modified and additional parts:
# Copyright (c) 2019 National Institute of Information and Communications Technology.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from dictionary_with_bert import DictionaryWithBert


@register_task('translation_with_bert')
class TranslationWithBertTask(TranslationTask):
    """
    Translation task using pretrained BERT encoders.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--bert-model', type=str, metavar='DIR', required=True,
                            help='path to the BERT model')
        parser.add_argument('--fine-tuning', action='store_true',
                            help='if set, the BERT model will be tuned')
        parser.set_defaults(left_pad_source=False)

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        if hasattr(args, 'model_overrides'):
            model_overrides = eval(args.model_overrides)
            model_overrides['bert_model'] = args.bert_model
            args.model_overrides = "{}".format(model_overrides)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task. (e.g., load dictionaries).
        The class of the source dictionary is changed to that for the BERT encoder.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        task = super().setup_task(args, **kwargs)
        task.src_dict = DictionaryWithBert(task.src_dict)
        return task