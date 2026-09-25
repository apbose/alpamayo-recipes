# Shortcut Models reference fixture

`shortcut_models/targets_shortcut.py` is the upstream reference function from
[kvfrans/shortcut-models at 601004348667094e1b71f30942199759412d4432](https://github.com/kvfrans/shortcut-models/blob/601004348667094e1b71f30942199759412d4432/targets_shortcut.py).
It is included only for offline numerical-parity tests, not imported by training.
The tests substitute NumPy-backed random draws so the same inputs can be fed to
our PyTorch implementation. JAX installation or another local checkout is not
required. Both the paper's 1/4 allocation and the repository's 1/8 default are tested.

The only file-normalization change is a final newline. SHA-256:

- Original: `adb2ac83febc1de012a7cdb713b9117d53616473ccd38f1104fd2475d8b1abc3`.
- Checked-in fixture: `903543035d1aa641a16a5540d841c075a649310549e1795bf0eaabcc5784976f`.

Copyright (c) 2024 Kevin Frans. The original MIT license is preserved in
[SHORTCUT_MODELS_LICENSE.txt](SHORTCUT_MODELS_LICENSE.txt). The target helpers
and sinusoidal encoding port retain this attribution; Alpamayo-specific code
uses this repository's Apache-2.0 license.
