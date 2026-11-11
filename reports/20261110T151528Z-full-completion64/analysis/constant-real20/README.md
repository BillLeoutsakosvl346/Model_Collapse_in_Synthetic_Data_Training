# Constant learning rate on repeated real data

Original pretrained OPT-125M, twenty continued epochs on the identical first 10,000 FineWeb pairs. AdamW, optimizer/RNG carry, continuation-only 64-token loss and fixed evaluation are preserved. Learning rate is 5e-6 throughout all 1,580 updates. There is no reset between epochs.

Test perplexity: 29.17618 initially, minimum 28.34749 at epoch 2, final 30.80345.

Fixed training NLL: 3.41101 to 2.72309.

Falling training loss with rising held-out loss is consistent with overfitting. This is a single-seed finite trajectory; perplexity alone does not uniquely diagnose synthetic-data collapse. The historical comparison has ten epochs, a decaying learning rate and RTX PRO 4000 hardware; the new run has twenty epochs and RTX 2000 Ada hardware. These are recorded differences.

All twenty checkpoints and optimization states remain in private backups. The terminal stage was cold-restored and re-evaluated on the GPU; this does not claim cold restoration of all twenty bulk stages. GPU compute is stopped; the working disk and backup volume remain retained.

Conservative task cost at this receipt: USD 0.8549; retained new storage approximately USD 0.53/day.

Reproduce from the recovered raw run with `python reports/20261110T151528Z-full-completion64/analysis/constant-real20/analysis_source.py --out reports/20261110T151528Z-full-completion64/analysis/reproduction-NEW`. Existing evidence is never overwritten.
