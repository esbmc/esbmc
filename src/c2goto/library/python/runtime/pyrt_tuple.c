#include "pyrt.h"

int64_t pyrt_tuple_length(PyRtObject *o);
PyRtObject *pyrt_tuple_item(PyRtObject *o, int64_t i);
PyRtObject *pyrt_tuple_richcompare(PyRtObject *a, PyRtObject *b, int op);

/* No sq_ass_item: a tuple cannot be assigned into. */
PyRtSequenceMethods pyrt_tuple_as_sequence = {
  .sq_length = pyrt_tuple_length,
  .sq_item = pyrt_tuple_item};

PyRtTypeObject PyRtTuple_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "tuple",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_tuple_as_sequence,
  .tp_richcompare = pyrt_tuple_richcompare};

PyRtObject *pyrt_tuple_new(void)
{
  PyRtTupleObject *t = __ESBMC_alloca(sizeof(PyRtTupleObject));
  t->ob_type = &PyRtTuple_Type;
  t->size = 0;
  return (PyRtObject *)t;
}

/* Only the frontend calls this, and only while the literal is being built:
 * once it is bound to a name the tuple never changes again. */
void pyrt_tuple_append(PyRtObject *o, PyRtObject *value)
{
  PyRtTupleObject *t = (PyRtTupleObject *)o;
  __ESBMC_assert(
    t->size < PYRT_TUPLE_CAPACITY, "pyrt: tuple exceeds the model capacity");
  t->items[t->size++] = value;
}

int64_t pyrt_tuple_length(PyRtObject *o)
{
  return ((PyRtTupleObject *)o)->size;
}

PyRtObject *pyrt_tuple_item(PyRtObject *o, int64_t i)
{
  PyRtTupleObject *t = (PyRtTupleObject *)o;
  if (i < 0 || i >= t->size)
    PYRT_RAISE("IndexError: tuple index out of range");
  return t->items[i];
}

/* Equality is element-wise, as in CPython. Ordering is left to the caller's
 * NotImplemented handling rather than modelled. */
PyRtObject *pyrt_tuple_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (a->ob_type != &PyRtTuple_Type || b->ob_type != &PyRtTuple_Type)
    return &pyrt_NotImplemented;
  if (op != Py_EQ && op != Py_NE)
    return &pyrt_NotImplemented;
  PyRtTupleObject *x = (PyRtTupleObject *)a;
  PyRtTupleObject *y = (PyRtTupleObject *)b;
  bool equal = x->size == y->size;
  for (int64_t i = 0; i < PYRT_TUPLE_CAPACITY && i < x->size && equal; ++i)
    if (!pyrt_key_equal(x->items[i], y->items[i]))
      equal = false;
  return pyrt_bool_from(op == Py_EQ ? equal : !equal);
}

/* `a, b = t` unpacks by length, so a shape mismatch is a ValueError rather
 * than an out-of-range read. Works for any sequence, not just a tuple. */
void pyrt_unpack_check(PyRtObject *o, int64_t expected)
{
  int64_t length = pyrt_iter_length(o);
  if (length < expected)
    PYRT_RAISE("ValueError: not enough values to unpack");
  if (length > expected)
    PYRT_RAISE("ValueError: too many values to unpack");
}
