"""Differential tests validating the range model against CPython's range().

models/range.py provides the two helpers ESBMC lowers a Python ``range`` loop
into: ``ESBMC_range_next_`` (advance the cursor) and ``ESBMC_range_has_next_``
(loop guard). Driving them as CPython does should reproduce ``list(range(...))``.
"""

import importlib.util
import os

from hypothesis import given, settings, strategies as st


def load_range_model():
    """Load models/range.py as a standalone module."""
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "python-frontend", "models", "range.py"
        )
    )
    spec = importlib.util.spec_from_file_location("range_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model = load_range_model()
has_next = _model.ESBMC_range_has_next_
next_ = _model.ESBMC_range_next_


def model_range(start, stop, step):
    """Reproduce list(range(start, stop, step)) using the model helpers."""
    result = []
    curr = start
    # Safety cap: the bounded strategies below keep ranges small; the cap only
    # guards against an unexpected non-terminating guard, never trims a valid range.
    while has_next(curr, stop, step):
        result.append(curr)
        curr = next_(curr, step)
        if len(result) > 100000:
            raise AssertionError("range model did not terminate")
    return result


bounded = st.integers(min_value=-200, max_value=200)
nonzero_step = st.integers(min_value=-50, max_value=50).filter(lambda s: s != 0)


class TestRangeIteration:
    @given(start=bounded, stop=bounded, step=nonzero_step)
    @settings(max_examples=400)
    def test_matches_cpython(self, start, stop, step):
        assert model_range(start, stop, step) == list(range(start, stop, step))

    @given(start=bounded, stop=bounded)
    def test_positive_step_one(self, start, stop):
        assert model_range(start, stop, 1) == list(range(start, stop, 1))

    @given(start=bounded, stop=bounded)
    def test_negative_step_one(self, start, stop):
        assert model_range(start, stop, -1) == list(range(start, stop, -1))


class TestRangeGuard:
    @given(curr=bounded, end=bounded, step=nonzero_step)
    @settings(max_examples=400)
    def test_has_next_matches_loop_condition(self, curr, end, step):
        # CPython keeps yielding while curr is on the correct side of end.
        expected = curr < end if step > 0 else curr > end
        assert has_next(curr, end, step) == expected

    @given(curr=bounded, end=bounded)
    def test_step_zero_is_never_runnable(self, curr, end):
        # step == 0 is invalid in CPython (ValueError); the model reports "no next".
        assert has_next(curr, end, 0) is False


class TestRangeNext:
    @given(curr=bounded, step=st.integers(min_value=-50, max_value=50))
    def test_next_advances_by_step(self, curr, step):
        assert next_(curr, step) == curr + step
