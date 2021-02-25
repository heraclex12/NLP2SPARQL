# The original part of the fairseq: Copyright (c) Facebook, Inc. and its affiliates.
# The modified and additional parts:
# Copyright (c) 2019 National Institute of Information and Communications Technology.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import BertModel

from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
)
from fairseq.models.transformer import (
    TransformerModel,
    EncoderOut,
    base_architecture,
)


@register_model('transformer_with_pretrained_bert')
class TransformerWithBertModel(TransformerModel):
    """
    Transformer model with the pretrained BERT encoder.
    This class replaces the Transformer encoder with that of BERT.
    See `"Recycling a Pre-trained BERT Encoder for Neural Machine
    Translation" (Imamura and Sumita, 2019)
    <https://www.aclweb.org/anthology/D19-5603/>`_.
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    #def __init__(self, args, encoder, decoder):
    #    super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = super().build_model(args, task)
        model.fine_tuning = args.fine_tuning
        return model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerWithBertEncoder(args, src_dict, embed_tokens)


    def train(self, mode=True):
        if self.fine_tuning:
            self.encoder.bert_model.train(mode)
            self.decoder.train(mode)
        else:
            self.encoder.bert_model.eval()
            self.decoder.train(mode)

    def eval(self):
        self.encoder.bert_model.eval()
        self.decoder.eval()


class TransformerWithBertEncoder(FairseqEncoder):
    """
    Transformer encoder acquired from the pretrained BERT encoder.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.padding_idx = embed_tokens.padding_idx
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.fine_tuning = args.fine_tuning
        self.bert_model = BertModel.from_pretrained(args.bert_model,
                                                    output_hidden_states=True, return_dict=False)

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x = None
        if not self.fine_tuning:
            with torch.no_grad():
                encoder_padding_mask = src_tokens.eq(self.padding_idx)
                attention_mask = src_tokens.ne(self.padding_idx).long()
                x, _, layer_outputs = self.bert_model(input_ids=src_tokens,
                                                      attention_mask=attention_mask)
                x = x.transpose(0, 1).detach()
                encoder_embedding = layer_outputs[0].detach()
                encoder_states = None
                if return_all_hiddens:
                    encoder_states = [layer_outputs[i].transpose(0, 1).detach()
                                      for i in range(1, len(layer_outputs))]

        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            attention_mask = src_tokens.ne(self.padding_idx).long()
            x, _, layer_outputs = self.bert_model(src_tokens,
                                                  attention_mask=attention_mask)
            x = x.transpose(0, 1)
            encoder_embedding = layer_outputs[0]
            encoder_states = None
            if return_all_hiddens:
                encoder_states = [layer_outputs[i].transpose(0, 1)
                                  for i in range(1, len(layer_outputs))]

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )



@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert')
def transformer_with_pretrained_bert(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    base_architecture(args)


@register_model_architecture('transformer_with_pretrained_bert',
                             'transformer_with_pretrained_bert_large')
def transformer_with_pretrained_bert_large(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)