"""The diagnostic must distinguish text error from surrounding background damage."""
from pathlib import Path
import runpy

import numpy as np


def test_truth_score_separates_errors():
    score = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'scripts/verify_reconstruction_truth.py'))['score']
    truth = np.full((1, 1, 3, 3), 100, np.uint8)
    dirty = truth.copy()
    dirty[0, 0, 0] = 200
    glyph = np.array([[[True, False, False]]])
    mask = np.array([[[True, True, False]]])
    perfect = score(truth, truth, dirty, glyph, mask)
    assert perfect['glyph_mae'] == perfect['background_mae'] == 0
    unchanged = score(dirty, truth, dirty, glyph, mask)
    assert unchanged['glyph_mae'] == 100 and unchanged['subtitle_error_projection'] == 1
    damaged = truth.copy()
    damaged[0, 0, 1] = 120
    result = score(damaged, truth, dirty, glyph, mask)
    assert result['glyph_mae'] == 0 and result['background_mae'] == 20
    assert result['outside_equal']
    damaged[0, 0, 2] = 0
    assert not score(damaged, truth, dirty, glyph, mask)['outside_equal']
