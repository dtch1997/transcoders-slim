import torch
import pytest
from transcoders_slim.transcoder import Transcoder # noqa
from transcoders_slim.load_pretrained import load_pretrained, FILENAMES

@pytest.mark.xfail
def test_load_pretrained():
    transcoders = load_pretrained(FILENAMES[:1])
    assert len(transcoders) == 1

    # test the transcoder
    transcoder = transcoders[FILENAMES[0]]
    d_in = transcoder.d_in
    d_tr = transcoder.d_sae
    seq_len = 32
    tr_in = torch.zeros(1, seq_len, d_in)
    tr_out, tr_hid = transcoder(input)[:2]

    assert tr_out.shape == tr_in.shape
    assert tr_hid.shape[-1] == d_tr