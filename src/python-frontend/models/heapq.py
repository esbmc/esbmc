# pylint: disable=unused-argument,useless-return,unidiomatic-typecheck
# Operational-model stubs for heapq. Argument names on heapify (whose
# body is intentionally abstract) are part of the API contract matched
# by ESBMC's converter. The explicit `return None` keeps the function's
# return-type analysis aligned with the `-> None` annotation. The
# `type(item) is tuple` test is intentional: in a closed-world FV model
# the strict identity check is more correct than isinstance(), which
# would also accept tuple subclasses with potentially different
# semantics under verification.
from typing import Any


def _heap_key(item: Any):
    if type(item) is tuple:
        return item[0]
    return item


def heapify(heap: list[Any]) -> None:
    return None


def heappush(heap: list[Any], item: Any) -> None:
    heap.append(item)
    return None


def heappop(heap: list[Any]) -> Any:
    best_index = 0
    i = 1
    while i < len(heap):
        if _heap_key(heap[i]) < _heap_key(heap[best_index]):
            best_index = i
        i = i + 1
    return heap.pop(best_index)


def heappushpop(heap: list[Any], item: Any) -> Any:
    heap.append(item)
    return heappop(heap)
