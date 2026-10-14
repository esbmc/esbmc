# HumanEval/17 — negative variant: a wrong expected result must NOT verify.
# Exercises the same runtime-split + comprehension + dict-lookup path as
# humaneval_17, guarding against a vacuous (always-SUCCESSFUL) pass.

from typing import List


def parse_music(music_string: str) -> List[int]:
    note_map = {'o': 4, 'o|': 2, '.|': 1}
    return [note_map[x] for x in music_string.split(' ') if x]


if __name__ == "__main__":
    # 'o o o o' parses to [4, 4, 4, 4]; the dropped element makes this false.
    assert parse_music('o o o o') == [4, 4, 4]
