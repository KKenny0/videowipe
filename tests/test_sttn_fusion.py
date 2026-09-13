"""Every temporal prediction contributes equally, including overlapping windows."""
import itertools
import numpy as np
from videowipe.inpainters.sttn import blend_frames, _process_segment


def test_temporal_fusion_is_order_independent_with_repeated_frame_ids():
    for values in itertools.permutations([0., 30., 90.]):
        output = np.zeros((2, 1, 1, 3), np.float32)
        counts = np.zeros(2, np.int32)
        blend_frames(output, np.array(values, np.float32).reshape(3, 1, 1, 1), [0, 0, 0], counts)
        np.testing.assert_allclose(output[0] / counts[0], 40.)
        assert counts.tolist() == [3, 0]
        assert not output[1].any()


def test_segment_fusion_counts_all_three_windows():
    class Backend:
        def preprocess(self, frames): return np.asarray(frames)
        def encode(self, frames): return frames
        def transform(self, frames):
            self.calls += 1
            return np.full_like(frames, self.calls * 30)
        def decode(self, frames): return frames
        calls = 0
    result = _process_segment([np.zeros((1, 1, 3), np.float32) for _ in range(25)],
                              Backend(), 1, 1, 5, 5)
    assert len(result) == 25 and all(frame is not None for frame in result)
    assert np.all(result[5] == 60)  # Windows centered at 0, 5, 10.
    assert np.all(result[2] == 45)  # Two-window result stays unchanged.
