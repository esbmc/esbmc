import random


def list_comp(actions, condition):
    result = []
    index = 0
    while index < len(actions):
        candidate = actions[index]
        if condition(candidate):
            result.append(candidate)
        index += 1
    return result


class Action:

    def pre(self) -> bool:
        raise NotImplementedError

    def act(self) -> None:
        raise NotImplementedError


class Down(Action):

    def pre(self) -> bool:
        return counter > 0

    def act(self) -> None:
        global counter
        counter -= 1
        assert counter >= 0
        print(f"counting down: {counter}")


class Up(Action):

    def pre(self) -> bool:
        return counter < 1

    def act(self) -> None:
        global counter
        counter += 1
        assert counter <= 1
        print(f"counting up: {counter}")


def main() -> None:
    actions = [Down(), Up()]
    while True:
        enabled_actions = list_comp(actions, lambda candidate: candidate.pre())

        if not enabled_actions:
            break

        length = len(enabled_actions)
        idx = random.randint(0, length - 1)
        print(f"length={length} action={idx}")
        chosen: Action = enabled_actions[idx]
        chosen.act()


counter: int = 1

main()
