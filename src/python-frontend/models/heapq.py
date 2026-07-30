# Operational model for heapq, ported from CPython's Lib/heapq.py.
#
# The sift routines reproduce CPython's array layout element-for-element, so
# heap[0], the pop order and tie-breaking all agree with the reference
# implementation for any input list -- including lists that do not satisfy the
# heap invariant. A model that merely scanned for the minimum would be unsound:
# heappop() returns heap[0], which is the minimum only once the invariant
# holds, and the invariant cannot be checked statically.
#
# The element type must be spelled out: without one the frontend infers elements
# to be lists and mis-compares them, and `list[Any]` makes index assignment
# through an imported module report a spurious violation. Hence `list[int]`.
#
# Only int elements are fully modelled.
#   float: sound but loud. heapify() and index reads are exact, while the
#     value-returning operations return an unconstrained result, so ESBMC
#     reports FAILED on assertions that hold in CPython. No float program was
#     found for which ESBMC reports SUCCESSFUL while CPython's assertion fails.
#     Pinned by regression/python/github_5931_float_knownbug.
#   str: sound but loud, in both directions. Pinned by github_5931_str_knownbug.
#   tuple: modelled. The `list[int]` annotation above is overridden by the
#     caller's actual element type (GitHub #5936), so a tuple-keyed heap --
#     `heappush(h, (priority, task))`, the canonical idiom -- comes out with the
#     right layout. Pinned by github_5931_tuple_fail. One heap element type per
#     program: the frontend emits one GOTO function per Python function, so a
#     program that heapifies tuples *and* pops from an int heap leaves this
#     annotation in place and stays unsound (github_5931_mixed_heap_knownbug_fail).


def _siftdown(heap: list[int], startpos: int, pos: int) -> None:
    """Move heap[pos] toward startpos until its parent is no larger."""
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) // 2
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
        else:
            break
    heap[pos] = newitem


def _siftup(heap: list[int], pos: int) -> None:
    """Bubble the smaller child up to heap[pos], then sift the oddball down."""
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos:
            left = heap[childpos]
            right = heap[rightpos]
            if not left < right:
                childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


def heapify(heap: list[int]) -> None:
    """Transform the list into a heap, in-place."""
    i = len(heap) // 2 - 1
    while i >= 0:
        _siftup(heap, i)
        i = i - 1


def heappush(heap: list[int], item: int) -> None:
    """Push item onto the heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)


def heappop(heap: list[int]) -> int:
    """Pop and return the smallest item, maintaining the heap invariant."""
    lastelt = heap.pop()
    if len(heap) == 0:
        return lastelt
    returnitem = heap[0]
    heap[0] = lastelt
    _siftup(heap, 0)
    return returnitem


def heapreplace(heap: list[int], item: int) -> int:
    """Pop and return the smallest item, then push item; heap size is unchanged."""
    returnitem = heap[0]
    heap[0] = item
    _siftup(heap, 0)
    return returnitem


def heappushpop(heap: list[int], item: int) -> int:
    """Push item on the heap, then pop and return the smallest item."""
    if len(heap) > 0:
        smallest = heap[0]
        if smallest < item:
            heap[0] = item
            _siftup(heap, 0)
            return smallest
    return item
