"""Legacy scientific byte conventions; changing these changes recorded identities."""
import hashlib

import numpy as np


def token_hash(values):
    """Hash C-order native int32 token bytes, matching all historical token seals.

    Prompt/raw-file hashes and composite token/mask identities are separate
    conventions and must not be substituted with this function.
    """
    return hashlib.sha256(np.asarray(values, dtype=np.int32).tobytes()).hexdigest()
