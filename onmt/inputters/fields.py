"""Module for build dynamic fields."""
from collections import Counter, defaultdict
import torch
from onmt.utils.logging import logger
from onmt.utils.misc import check_path
from onmt.inputters.inputter import get_fields, _load_vocab, \
    _build_fields_vocab


def _get_dynamic_fields(opts):
    # NOTE: not support tgt feats yet
    tgt_feats = None
    with_align = hasattr(opts, 'lambda_align') and opts.lambda_align > 0.0
    fields = get_fields('text', opts.src_feats_vocab, tgt_feats,
                        dynamic_dict=opts.copy_attn,
                        src_truncate=opts.src_seq_length_trunc,
                        tgt_truncate=opts.tgt_seq_length_trunc,
                        with_align=with_align,
                        data_task=opts.data_task, pad='<PAD>')

    return fields


def build_dynamic_fields(opts, src_specials=None, tgt_specials=None):
    """Build fields for dynamic, including load & build vocab."""
    fields = _get_dynamic_fields(opts)

    counters = defaultdict(Counter)
    logger.info("Loading vocab from text file...")

    _src_vocab, _src_vocab_size = _load_vocab(
        opts.src_vocab, 'src', counters,
        min_freq=opts.src_words_min_frequency)

    if opts.src_feats_vocab:
        for feat_name, filepath in opts.src_feats_vocab.items():
            _, _ = _load_vocab(
                filepath, feat_name, counters,
                min_freq=0)

    if opts.tgt_vocab:
        _tgt_vocab, _tgt_vocab_size = _load_vocab(
            opts.tgt_vocab, 'tgt', counters,
            min_freq=opts.tgt_words_min_frequency)
    elif opts.share_vocab:
        logger.info("Sharing src vocab to tgt...")
        counters['tgt'] = counters['src']
    else:
        raise ValueError("-tgt_vocab should be specified if not share_vocab.")

    if opts.feat_merge in ['sharemlp', 'sharemlplin']:
        shared_feat_count = sum([counters[feat_name] for feat_name in opts.src_feats_vocab.keys()], Counter())
        logger.info(f"Building a shared feature vocab... size: {len(shared_feat_count)}")
        # counters['shared_src_feat'] = shared_feat_count
        for feat_name in opts.src_feats_vocab.keys():
            counters[feat_name] = shared_feat_count

    logger.info("Building fields with vocab in counters...")
    fields = _build_fields_vocab(
        fields, counters, 'text', opts.share_vocab,
        opts.vocab_size_multiple,
        opts.src_vocab_size, opts.src_words_min_frequency,
        opts.tgt_vocab_size, opts.tgt_words_min_frequency,
        src_specials=src_specials, tgt_specials=tgt_specials)

    return fields


def get_vocabs(fields):
    """Get a dict contain src & tgt vocab extracted from fields."""
    src_vocab = fields['src'].base_field.vocab
    tgt_vocab = fields['tgt'].base_field.vocab
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    return vocabs


def save_fields(fields, save_data, overwrite=True):
    """Dump `fields` object."""
    fields_path = "{}.vocab.pt".format(save_data)
    check_path(fields_path, exist_ok=overwrite, log=logger.warning)
    logger.info(f"Saving fields to {fields_path}...")
    torch.save(fields, fields_path)


def load_fields(save_data, checkpoint=None):
    """Load dumped fields object from `save_data` or `checkpoint` if any."""
    if checkpoint is not None:
        logger.info("Loading fields from checkpoint...")
        fields = checkpoint['vocab']
    else:
        fields_path = "{}.vocab.pt".format(save_data)
        logger.info(f"Loading fields from {fields_path}...")
        fields = torch.load(fields_path)
    return fields
