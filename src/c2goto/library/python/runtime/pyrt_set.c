#include "pyrt.h"

int64_t pyrt_set_length(PyRtObject *o);
PyRtObject *pyrt_set_item(PyRtObject *o, int64_t i);
PyRtObject *pyrt_set_richcompare(PyRtObject *a, PyRtObject *b, int op);

/* No sq_ass_item: a member is added or discarded, never assigned by index. */
PyRtSequenceMethods pyrt_set_as_sequence = {
  .sq_length = pyrt_set_length,
  .sq_item = pyrt_set_item};

PyRtTypeObject PyRtSet_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "set",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_set_as_sequence,
  .tp_richcompare = pyrt_set_richcompare};

PyRtObject *pyrt_set_new(void)
{
  PyRtSetObject *s = __ESBMC_alloca(sizeof(PyRtSetObject));
  s->ob_type = &PyRtSet_Type;
  s->size = 0;
  return (PyRtObject *)s;
}

int64_t pyrt_set_find(PyRtSetObject *s, PyRtObject *value)
{
  for (int64_t i = 0; i < PYRT_SET_CAPACITY && i < s->size; ++i)
    if (pyrt_key_equal(s->items[i], value))
      return i;
  return -1;
}

/* Adding a member already present changes nothing, which is what makes the
 * literal `{1, 1}` a set of one. */
void pyrt_set_add(PyRtObject *o, PyRtObject *value)
{
  PyRtSetObject *s = (PyRtSetObject *)o;
  if (pyrt_set_find(s, value) >= 0)
    return;
  __ESBMC_assert(
    s->size < PYRT_SET_CAPACITY, "pyrt: set exceeds the model capacity");
  s->items[s->size++] = value;
}

/* discard() is silent about a member that is not there; remove() is not. */
void pyrt_set_discard(PyRtObject *o, PyRtObject *value)
{
  PyRtSetObject *s = (PyRtSetObject *)o;
  const int64_t at = pyrt_set_find(s, value);
  if (at < 0)
    return;
  for (int64_t i = 0; i < PYRT_SET_CAPACITY - 1; ++i)
    if (i >= at && i + 1 < s->size)
      s->items[i] = s->items[i + 1];
  s->size--;
}

void pyrt_set_remove(PyRtObject *o, PyRtObject *value)
{
  if (pyrt_set_find((PyRtSetObject *)o, value) < 0)
    PYRT_RAISE("KeyError");
  pyrt_set_discard(o, value);
}

int64_t pyrt_set_length(PyRtObject *o)
{
  return ((PyRtSetObject *)o)->size;
}

PyRtObject *pyrt_set_item(PyRtObject *o, int64_t i)
{
  PyRtSetObject *s = (PyRtSetObject *)o;
  if (i < 0 || i >= s->size)
    PYRT_RAISE("IndexError: set index out of range");
  return s->items[i];
}

/* Two sets are equal when each holds the other's members, whatever order they
 * were added in. */
PyRtObject *pyrt_set_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (a->ob_type != &PyRtSet_Type || b->ob_type != &PyRtSet_Type)
    return &pyrt_NotImplemented;
  if (op != Py_EQ && op != Py_NE)
    return &pyrt_NotImplemented;
  PyRtSetObject *x = (PyRtSetObject *)a;
  PyRtSetObject *y = (PyRtSetObject *)b;
  bool equal = x->size == y->size;
  for (int64_t i = 0; i < PYRT_SET_CAPACITY && i < x->size && equal; ++i)
    if (pyrt_set_find(y, x->items[i]) < 0)
      equal = false;
  return pyrt_bool_from(op == Py_EQ ? equal : !equal);
}
