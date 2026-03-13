"""
Runtime fix for XLNet positional encoding device placement.

HuggingFace Transformers' XLNetModel.relative_positional_encoding (confirmed
in v4.30.2) calls torch.arange() without a device= argument.  This forces the
sinusoidal positional embedding computation onto CPU every forward pass and
then copies the result to GPU -- adding up to hundreds of milliseconds of
overhead per step depending on sequence length and hardware.

The fix adds device= to all torch.arange calls so that the entire computation
stays on whatever device the model weights reside on.

Usage
-----
Import and call before training:

    from tools.xlnet_pos_encoding_fix import patch_xlnet_positional_encoding
    patch_xlnet_positional_encoding()

Or as a script:

    python -m tools.xlnet_pos_encoding_fix   # patches and exits

The patch is idempotent -- calling it multiple times is safe.
"""

import torch

_PATCHED = False


def patch_xlnet_positional_encoding():
    """Monkey-patch XLNetModel so positional encoding runs on GPU."""
    global _PATCHED
    if _PATCHED:
        return

    import transformers.models.xlnet.modeling_xlnet as _xlnet

    @staticmethod
    def _positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat(
            [torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1
        )
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)
        return pos_emb

    def _relative_positional_encoding(self, qlen, klen, bsz=None):
        device = self.word_embedding.weight.device

        freq_seq = torch.arange(
            0, self.d_model, 2.0, dtype=torch.float, device=device
        )
        inv_freq = 1.0 / torch.pow(10000, freq_seq / self.d_model)

        if self.attn_type == "bi":
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown attn_type {self.attn_type!r}")

        if self.bi_data:
            fwd_pos_seq = torch.arange(
                beg, end, -1.0, dtype=torch.float, device=device
            )
            bwd_pos_seq = torch.arange(
                -beg, -end, 1.0, dtype=torch.float, device=device
            )
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(
                    fwd_pos_seq, inv_freq, bsz // 2
                )
                bwd_pos_emb = self.positional_embedding(
                    bwd_pos_seq, inv_freq, bsz // 2
                )
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0, device=device)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    _xlnet.XLNetModel.positional_embedding = _positional_embedding
    _xlnet.XLNetModel.relative_positional_encoding = _relative_positional_encoding

    _PATCHED = True


if __name__ == "__main__":
    patch_xlnet_positional_encoding()
    print("XLNet positional encoding GPU fix applied.")
