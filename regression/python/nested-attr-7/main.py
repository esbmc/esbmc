# Test case 7: Nested attribute with method returning object (chaining)
class Node:

    def __init__(self, value: int) -> None:
        self.value: int = value

    def get_value(self) -> int:
        return self.value


class ListBuilder:

    def __init__(self) -> None:
        self.head: Node = Node(1)

    def build(self) -> Node:
        return self.head


class Manager:

    def __init__(self) -> None:
        self.builder: ListBuilder = ListBuilder()

    def get_head_value(self) -> int:
        # Chain: self.builder.build() returns Node, then call get_value()
        result = self.builder.build().get_value()
        return result


manager = Manager()
value: int = manager.get_head_value()
assert value == 1
