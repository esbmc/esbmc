#include "pyrt.h"

int64_t pyrt_list_length(PyRtObject *o);
PyRtObject *pyrt_list_item(PyRtObject *o, int64_t i);
void pyrt_list_ass_item(PyRtObject *o, int64_t i, PyRtObject *value);

PyRtSequenceMethods pyrt_list_as_sequence = {
  .sq_length = pyrt_list_length,
  .sq_item = pyrt_list_item,
  .sq_ass_item = pyrt_list_ass_item};

PyRtObject *pyrt_list_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtTypeObject PyRtList_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "list",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_list_as_sequence,
  .tp_richcompare = pyrt_list_richcompare};

/* Element-wise, as in CPython: two lists holding equal items are equal
 * whether or not they are the same object. Ordering is left to the caller's
 * NotImplemented handling rather than modelled. */
PyRtObject *pyrt_list_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (a->ob_type != &PyRtList_Type || b->ob_type != &PyRtList_Type)
    return &pyrt_NotImplemented;
  if (op != Py_EQ && op != Py_NE)
    return &pyrt_NotImplemented;
  PyRtListObject *x = (PyRtListObject *)a;
  PyRtListObject *y = (PyRtListObject *)b;
  bool equal = x->size == y->size;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < x->size && equal; ++i)
    if (!pyrt_key_equal(x->items[i], y->items[i]))
      equal = false;
  return pyrt_bool_from(op == Py_EQ ? equal : !equal);
}

PyRtObject *pyrt_list_new(void)
{
  PyRtListObject *l = __ESBMC_alloca(sizeof(PyRtListObject));
  l->ob_type = &PyRtList_Type;
  l->size = 0;
  return (PyRtObject *)l;
}

int64_t pyrt_list_length(PyRtObject *o)
{
  return ((PyRtListObject *)o)->size;
}

PyRtObject *pyrt_list_item(PyRtObject *o, int64_t i)
{
  PyRtListObject *l = (PyRtListObject *)o;
  if (i < 0 || i >= l->size)
    PYRT_RAISE("IndexError: list index out of range");
  return l->items[i];
}

void pyrt_list_ass_item(PyRtObject *o, int64_t i, PyRtObject *value)
{
  PyRtListObject *l = (PyRtListObject *)o;
  if (i < 0 || i >= l->size)
    PYRT_RAISE("IndexError: list assignment index out of range");
  l->items[i] = value;
}

void pyrt_list_append(PyRtObject *o, PyRtObject *value)
{
  if (o->ob_type != &PyRtList_Type)
    PYRT_RAISE("AttributeError: object has no attribute 'append'");
  PyRtListObject *l = (PyRtListObject *)o;
  __ESBMC_assert(
    l->size < PYRT_LIST_CAPACITY, "pyrt: list exceeds the model capacity");
  l->items[l->size++] = value;
}

static void pyrt_list_remove_at(PyRtListObject *l, int64_t at)
{
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY - 1; ++i)
    if (i >= at && i + 1 < l->size)
      l->items[i] = l->items[i + 1];
  l->size--;
}

int64_t pyrt_list_find(PyRtListObject *l, PyRtObject *value)
{
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < l->size; ++i)
    if (pyrt_key_equal(l->items[i], value))
      return i;
  return -1;
}

/* index == 0 is pop()'s no-argument form, which takes the last item. */
PyRtObject *pyrt_listmeth_pop(PyRtObject *o, PyRtObject *index)
{
  PyRtListObject *l = (PyRtListObject *)o;
  if (l->size == 0)
    PYRT_RAISE("IndexError: pop from empty list");
  int64_t at = index ? pyrt_as_index(index) : l->size - 1;
  if (at < 0)
    at += l->size;
  if (at < 0 || at >= l->size)
    PYRT_RAISE("IndexError: pop index out of range");
  PyRtObject *value = l->items[at];
  pyrt_list_remove_at(l, at);
  return value;
}

void pyrt_listmeth_extend(PyRtObject *o, PyRtObject *iterable)
{
  int64_t count = pyrt_iter_length(iterable);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < count; ++i)
    pyrt_list_append(o, pyrt_iter_item(iterable, i));
}

PyRtObject *pyrt_listmeth_index(PyRtObject *o, PyRtObject *value)
{
  int64_t at = pyrt_list_find((PyRtListObject *)o, value);
  if (at < 0)
    PYRT_RAISE("ValueError: value is not in list");
  return pyrt_long_from(at);
}

/* CPython clamps an out-of-range insertion point rather than raising. */
void pyrt_listmeth_insert(PyRtObject *o, PyRtObject *index, PyRtObject *value)
{
  PyRtListObject *l = (PyRtListObject *)o;
  __ESBMC_assert(
    l->size < PYRT_LIST_CAPACITY, "pyrt: list exceeds the model capacity");
  int64_t at = pyrt_as_index(index);
  if (at < 0)
    at += l->size;
  if (at < 0)
    at = 0;
  if (at > l->size)
    at = l->size;
  #pragma unroll
  for (int64_t i = PYRT_LIST_CAPACITY - 1; i > 0; --i)
    if (i <= l->size && i > at)
      l->items[i] = l->items[i - 1];
  l->items[at] = value;
  l->size++;
}

void pyrt_listmeth_remove(PyRtObject *o, PyRtObject *value)
{
  PyRtListObject *l = (PyRtListObject *)o;
  int64_t at = pyrt_list_find(l, value);
  if (at < 0)
    PYRT_RAISE("ValueError: list.remove(x): x not in list");
  pyrt_list_remove_at(l, at);
}

PyRtObject *pyrt_listmeth_count(PyRtObject *o, PyRtObject *value)
{
  PyRtListObject *l = (PyRtListObject *)o;
  int64_t seen = 0;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < l->size; ++i)
    if (pyrt_key_equal(l->items[i], value))
      ++seen;
  return pyrt_long_from(seen);
}

void pyrt_listmeth_clear(PyRtObject *o)
{
  ((PyRtListObject *)o)->size = 0;
}

PyRtObject *pyrt_listmeth_copy(PyRtObject *o)
{
  PyRtListObject *l = (PyRtListObject *)o;
  PyRtObject *result = pyrt_list_new();
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < l->size; ++i)
    pyrt_list_append(result, l->items[i]);
  return result;
}

/* Insertion sort: stable, as CPython guarantees, and its shape is fixed by
 * the capacity so it unrolls like every other loop here. */
void pyrt_listmeth_sort(PyRtObject *o)
{
  PyRtListObject *l = (PyRtListObject *)o;
  #pragma unroll
  for (int64_t i = 1; i < PYRT_LIST_CAPACITY; ++i)
  {
    if (i >= l->size)
      break;
    PyRtObject *key = l->items[i];
    int64_t j = i - 1;
    #pragma unroll
    for (int64_t k = 0; k < PYRT_LIST_CAPACITY; ++k)
    {
      if (j < 0 || !pyrt_is_true(pyrt_richcompare(key, l->items[j], Py_LT)))
        break;
      l->items[j + 1] = l->items[j];
      --j;
    }
    l->items[j + 1] = key;
  }
}
