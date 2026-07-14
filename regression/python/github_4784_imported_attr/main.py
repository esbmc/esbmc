# Cross-module attribute-type inference (#4784): `ListNode` is defined in
# listnode.py and only ever constructed in this module. Inside a function the
# nested read `node.successor.value` must resolve via the inferred field type
# of an *imported* class. Before the fix the attribute stayed any_type() and
# this aborted with "Cannot resolve nested attribute".
from listnode import ListNode


def second_value(node):
    return node.successor.value


a = ListNode(1)
b = ListNode(2, a)
assert second_value(b) == 1
