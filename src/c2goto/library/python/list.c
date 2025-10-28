#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>


/*
===============================================================================
                    COMPREHENSIVE PYTHON LIST FEATURES OVERVIEW
===============================================================================

NOTE: There is no nice reference for Python like C has. We pretty much have to
go through the docs and pray. So I just asked some LLM to generate one for me.
This might be incomplete/wrong so always double check before implementing stuff.
Regardless... it seemed like a good roadmap. Feel free to delete/add missing
features.

Some links to use for reference while implementing:
[1] Python Standard Library docs
    (https://docs.python.org/3/library/stdtypes.html#list)
[2] CPython implementation
    (https://github.com/python/cpython/blob/main/Objects/listobject.c)

Implementation Notes: In [2], the solution that the authors
implemented was to construct a FAM. ESBMC, however, hates FAMs, so we
can do just as easily with a pointer. They also rely on Malloc and
Garbage Collection for memory management. Which we will skip.

So for ESBMC, a "Python list" is just a "Vector" which keeps an
infinite array and its length. For iterators and some of the
incompatible functions we could use function pointers, but we should
avoid them. Abstract Interpration does not like them and we could just
generate code on demand at symex for edge cases.

===============================================================================
                                BASIC PROPERTIES
===============================================================================
- [ ] Ordered collection of mutable, heterogeneous objects (arbitrary Python objects)
- [ ] Preserves insertion order
- [ ] Supports dynamic resizing (arbitrary growth/shrinkage)
- [ ] Zero-based indexing with negative index support

===============================================================================
                          CONSTRUCTORS & INITIALIZATION
===============================================================================
- [ ] List literal [a, b, ...]
- [ ] list() constructor (empty list)
- [ ] list(iterable) constructor from any iterable
- [ ] From specific iterables: tuple, set, dict.keys()/values()/items()
- [ ] From generator expressions, range(), map(), filter(), zip(), enumerate()
- [ ] List comprehension: [expr for x in iterable if condition]
- [ ] Multiplication/repetition: [val] * n (creates shallow copies)
- [ ] Unpacking syntax: [*iterable1, *iterable2, element, ...]
- [ ] Nested list literals: [[a, b], [c, d]]

===============================================================================
                           ELEMENT ACCESS & SLICING
===============================================================================
- [ ] Indexing: lst[i], supports negative indices (-1 is last element)
- [ ] Slicing with start/end: lst[start:end] (end exclusive)
- [ ] Slicing with step: lst[start:end:step] (supports negative step)
- [ ] Extended slicing: lst[:], lst[::2], lst[::-1] (reverse)
- [ ] Slice assignment: lst[start:end] = iterable (can change list size)
- [ ] Index assignment: lst[i] = value
- [ ] Deletion by index: del lst[i]
- [ ] Deletion by slice: del lst[start:end:step]
- [ ] Membership testing: x in lst, x not in lst
- [ ] Iteration: for x in lst, iter(lst), next() protocol

===============================================================================
                              MUTATING METHODS
===============================================================================
- [ ] append(x): Add element to end - O(1) amortized
- [ ] extend(iterable): Extend by appending all elements from iterable
- [ ] insert(i, x): Insert at position i, shifts elements right - O(n)
- [ ] remove(x): Remove first matching value, raises ValueError if not found
- [ ] pop([i]): Remove/return item at index (end if omitted) - O(1) for end
- [ ] clear(): Remove all items, equivalent to del lst[:]
- [ ] reverse(): In-place reversal of list - O(n)
- [ ] sort(*, key=None, reverse=False): Sort list in-place using Timsort

Sort method details:
- key: function taking one argument, returns comparison key
- reverse: boolean to reverse sort order
- Stable sort (maintains relative order of equal elements)
- Uses Timsort algorithm (hybrid stable sort)

===============================================================================
                            NON-MUTATING METHODS
===============================================================================
- [ ] count(x): Count occurrences of value x - O(n)
- [ ] index(x[, start[, end]]): Find index of first occurrence
      - Raises ValueError if not found
      - Optional start/end parameters limit search range
- [ ] copy(): Create shallow copy, equivalent to lst[:]

===============================================================================
                    SEQUENCE/BUILT-IN OPERATORS & FUNCTIONS
===============================================================================
Operators:
- [ ] + (concatenation): lst1 + lst2 creates new list
- [ ] * (repetition): lst * n or n * lst creates new list
- [ ] == != (comparison): elementwise comparison, returns bool
- [ ] <, >, <=, >= (lexicographical ordering): compares elements left-to-right
- [ ] += (in-place concatenation): equivalent to extend()
- [ ] *= (in-place repetition): extends list by repeating itself

Built-in Functions:
- [ ] len(lst): Return number of elements - O(1)
- [ ] min(lst), max(lst): Return smallest/largest element
- [ ] sum(lst, start=0): Sum of elements plus optional start value
- [ ] any(lst): True if any element is truthy
- [ ] all(lst): True if all elements are truthy (empty list returns True)
- [ ] sorted(lst, *, key=None, reverse=False): Return new sorted list
- [ ] reversed(lst): Return reverse iterator
- [ ] enumerate(lst, start=0): Return enumerate object with (index, value) pairs
- [ ] zip(lst1, lst2, ...): Combine multiple sequences element-wise
- [ ] map(func, lst): Apply function to each element
- [ ] filter(func, lst): Filter elements based on function
- [ ] list(iterable): Convert iterable to list
- [ ] Unpacking: *lst in function calls and other contexts

===============================================================================
                        SEQUENCE PROTOCOL METHODS
===============================================================================
Special methods that enable sequence behavior:
- [ ] __len__(): Called by len(), returns number of elements
- [ ] __getitem__(index): Called for lst[index], supports slicing
- [ ] __setitem__(index, value): Called for lst[index] = value
- [ ] __delitem__(index): Called for del lst[index]
- [ ] __contains__(item): Called for 'item in lst'
- [ ] __iter__(): Return iterator object for 'for' loops
- [ ] __reversed__(): Called by reversed(), should return reverse iterator
- [ ] __add__(other): Called for lst + other
- [ ] __mul__(other): Called for lst * n
- [ ] __iadd__(other): Called for lst += other (in-place add)
- [ ] __imul__(other): Called for lst *= n (in-place multiply)
- [ ] __eq__(other), __ne__(other): Equality comparison
- [ ] __lt__(other), __le__(other), __gt__(other), __ge__(other): Ordering

===============================================================================
                      ADVANCED BEHAVIORS & EDGE CASES
===============================================================================
- [ ] Lists are mutable; assignment creates aliases, not copies
- [ ] Slicing creates shallow copies (new list, same object references)
- [ ] Nested lists: modification affects all references
- [ ] List comprehensions create new scope in Python 3
- [ ] Iteration during modification can cause issues (use copy for safety)
- [ ] Empty list evaluates to False in boolean context
- [ ] List concatenation with += vs + behavior differences

===============================================================================
                         EXCEPTIONS & ERROR HANDLING
===============================================================================
- [ ] IndexError: List index out of range (access invalid positive/negative index)
- [ ] ValueError: Value not found (remove(), index() when item not present)
- [ ] TypeError: 
      - Wrong argument type for methods
      - Unsupported operand types for operators
      - Unhashable type as argument where not expected
- [ ] MemoryError: When system runs out of memory during operations
- [ ] OverflowError: When index or size exceeds system limits

===============================================================================
                           IMPLEMENTATION CHECKLIST
===============================================================================
Legend:
- [ ] Feature incomplete or unimplemented
- [P] Partially working (basic functionality)
- [X] Fully implemented (passes basic tests)
===============================================================================
*/

typedef struct
{
  const void *value; // data pointer
  size_t type_id;    // hashed type name
  size_t size;       // number of bytes in value
} Object;

typedef struct
{
  Object *items;
  size_t size; // elements in use
} List;

void __ESBMC_create_list()
{
  assert(0);
};



/* ---------- create ---------- */
static inline List *list_create(Object *backing)
{
  List *l = __ESBMC_alloca(sizeof(List));

  __ESBMC_assume(l != NULL);

  l->items = backing;
  l->size = 0;
  return l;
}

/* ---------- bounds check ---------- */
/*
static inline bool list_in_bounds(const List *l, size_t index)
{
  return index < l->size;
}
*/

static bool list_eq(const List *l1, const List *l2)
{
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  size_t i = 0;
  while (i < l1->size)
  {
    const Object *a = &l1->items[i];
    const Object *b = &l2->items[i];

    // Same address => element equal; keep checking the rest.
    if (a->value == b->value)
    {
      ++i;
      continue;
    }

    if (
      !a->value || !b->value || a->type_id != b->type_id ||
      a->size != b->size || memcmp(a->value, b->value, a->size) != 0)
    {
      return false;
    }

    ++i;
  }
  return true;
}

static inline size_t list_size(const List *l)
{
  return l ? l->size : 0;
}

/* ---------- getters ---------- */
static inline Object *list_at(List *l, size_t index)
{
  //return list_in_bounds(l, index) ? &l->items[index] : NULL;
  assert(index < l->size);
  return &l->items[index];
}

static inline const Object *list_cat(const List *l, size_t index)
{
  //  return list_in_bounds(l, index) ? &l->items[index] : NULL;
  assert(index < l->size);
  return &l->items[index];
}

static inline void *list_get_as(const List *l, size_t i, size_t expect_type)
{
  const Object *o = list_cat(l, i);
  return (o && o->type_id == expect_type) ? (void *)o->value : NULL;
}

/* ---------- push element ---------- */

/*
static inline bool list_push(List *l, const void *value, size_t type_id)
{
  l->items[l->size].value = value;
  l->items[l->size].type_id = type_id;
  l->size++;
  return true;
}
*/

static inline bool
list_push(List *l, const void *value, size_t type_id, size_t type_size)
{
  void *copied_value = __ESBMC_alloca(type_size);

  memcpy(copied_value, value, type_size);

  l->items[l->size].value = copied_value;
  l->items[l->size].type_id = type_id;
  l->items[l->size].size = type_size;
  l->size++;
  return true;
}

static inline bool list_push_object(List *l, Object *o)
{
  assert(l != NULL);
  assert(o != NULL);
  return list_push(l, o->value, o->type_id, o->size);
}

/* ---------- insert element at index ---------- */
static inline bool list_insert(
  List *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  // If index is beyond the end, just append
  if (index >= l->size)
    return list_push(l, value, type_id, type_size);

  // Make a copy of the value
  void *copied_value = __ESBMC_alloca(type_size);
  memcpy(copied_value, value, type_size);

  // Shift all elements from index onwards one position to the right
  size_t elements_to_shift = l->size - index;
  memmove(
    &l->items[index + 1], &l->items[index], elements_to_shift * sizeof(Object));

  // Insert the new element
  l->items[index].value = copied_value;
  l->items[index].type_id = type_id;
  l->items[index].size = type_size;
  l->size++;

  return true;
}

/* ---------- replace element ---------- */
static inline bool
list_replace(List *l, size_t index, const void *new_value, size_t type_id)
{
  if (index >= l->size)
    return false;
  l->items[index].value = new_value;
  l->items[index].type_id = type_id;
  return true;
}

/* ---------- pop / erase ---------- */
static inline bool list_pop(List *l)
{
  if (l->size == 0)
    return false;
  l->size--;
  return true;
}

static bool list_contains(
  const List *l,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  if (!l || !item)
    return false;

  size_t i = 0;
  while (i < l->size)
  {
    const Object *elem = &l->items[i];

    // Check if types and sizes match
    if (elem->type_id == item_type_id && elem->size == item_size)
    {
      // Compare the actual data
      if (elem->value == item || memcmp(elem->value, item, item_size) == 0)
        return true;
    }

    ++i;
  }
  return false;
}

/* ---------- extend list ---------- */
static inline void list_extend(List *l, const List *other)
{
  if (!l || !other)
    return;

  size_t i = 0;
  while (i < other->size)
  {
    const Object *elem = &other->items[i];

    void *copied_value = __ESBMC_alloca(elem->size);
    memcpy(copied_value, elem->value, elem->size);

    l->items[l->size].value = copied_value;
    l->items[l->size].type_id = elem->type_id;
    l->items[l->size].size = elem->size;
    l->size++;

    ++i;
  }
}
