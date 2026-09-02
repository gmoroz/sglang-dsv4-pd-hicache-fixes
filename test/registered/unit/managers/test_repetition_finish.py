import unittest
from array import array

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
    _has_repeated_tail,
)
from sglang.srt.sampling.sampling_params import SamplingParams


def _make_req() -> Req:
    return Req(
        rid="repetition",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=SamplingParams(max_new_tokens=1000, stop=[], stop_regex=[]),
        vocab_size=100000,
    )


def _run_tokens(tokens: list[int]) -> Req:
    req = _make_req()
    req.output_ids.extend(tokens)
    req.update_finish_state(new_accepted_len=len(tokens))
    return req


class TestRepetitionFinish(unittest.TestCase):
    def test_periods_from_four_to_sixty_four_repeat_five_times_finish(self):
        for period in (4, 7, 30, 64):
            pattern = list(range(period))
            req = _run_tokens(pattern * (8 if period == 4 else 5))
            self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
            self.assertEqual(req.finished_reason.to_json()["type"], "length")

    def test_three_repeats_do_not_finish(self):
        pattern = list(range(7))
        req = _run_tokens(pattern * 3 + list(range(100, 180)))
        self.assertFalse(req.finished())

    def test_four_repeats_do_not_finish_at_threshold_boundary(self):
        for period in (4, 7, 30, 64):
            pattern = list(range(period))
            self.assertFalse(_has_repeated_tail(array("q", pattern * 4)))

    def test_two_repeats_do_not_finish(self):
        pattern = list(range(30))
        self.assertFalse(_has_repeated_tail(array("q", pattern * 2)))

    def test_period_below_minimum_does_not_finish(self):
        pattern = list(range(3))
        self.assertFalse(_has_repeated_tail(array("q", pattern * 20)))

    def test_period_above_maximum_does_not_finish(self):
        pattern = list(range(65))
        self.assertFalse(_has_repeated_tail(array("q", pattern * 6)))

    def test_non_repeating_sequence_does_not_finish(self):
        req = _run_tokens(list(range(100)))
        self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
