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
