import random


def main() -> None:
    # choice picks an element actually in the sequence.
    v = random.choice([1, 2, 3])
    assert v in [1, 2, 3]

    # seed is a no-op: choice still observes the constrained domain.
    random.seed(42)
    w = random.choice([10, 20, 30])
    assert w in [10, 20, 30]

    # shuffle leaves length unchanged (under-approximation).
    xs = [1, 2, 3]
    random.shuffle(xs)
    assert len(xs) == 3

    # sample produces a list of length k from population.
    s = random.sample([7, 8, 9, 10], 2)
    assert len(s) == 2
main()
