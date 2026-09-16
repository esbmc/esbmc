#include "pyrt.h"

int64_t pyrt_str_length(PyRtObject *o);
PyRtObject *pyrt_str_item(PyRtObject *o, int64_t i);
PyRtObject *pyrt_str_concat(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_str_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtSequenceMethods pyrt_str_as_sequence = {
  .sq_length = pyrt_str_length,
  .sq_concat = pyrt_str_concat,
  .sq_item = pyrt_str_item};

PyRtTypeObject PyRtStr_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "str",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_str_as_sequence,
  .tp_richcompare = pyrt_str_richcompare};

bool pyrt_str_check(PyRtObject *o)
{
  return o->ob_type == &PyRtStr_Type;
}

PyRtObject *pyrt_str_new(const char *data, int64_t length)
{
  PyRtStrObject *s = __ESBMC_alloca(sizeof(PyRtStrObject));
  s->ob_type = &PyRtStr_Type;
  s->length = length;
  s->data = data;
  return (PyRtObject *)s;
}

int64_t pyrt_str_length(PyRtObject *o)
{
  return ((PyRtStrObject *)o)->length;
}

PyRtObject *pyrt_str_item(PyRtObject *o, int64_t i)
{
  PyRtStrObject *s = (PyRtStrObject *)o;
  if (i < 0 || i >= s->length)
    PYRT_RAISE("IndexError: string index out of range");
  char *one = __ESBMC_alloca(2);
  one[0] = s->data[i];
  one[1] = 0;
  return pyrt_str_new(one, 1);
}

PyRtObject *pyrt_str_concat(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_str_check(b))
    PYRT_RAISE("TypeError: can only concatenate str to str");
  PyRtStrObject *x = (PyRtStrObject *)a;
  PyRtStrObject *y = (PyRtStrObject *)b;
  int64_t total = x->length + y->length;
  if (total >= PYRT_STR_CAPACITY)
    PYRT_RAISE("pyrt: string longer than the model holds");

  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY);
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < x->length; ++i)
    buffer[i] = x->data[i];
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < y->length; ++i)
    buffer[x->length + i] = y->data[i];
  buffer[total] = 0;
  return pyrt_str_new(buffer, total);
}

/* Equality only: CPython also orders strings, which this model does not, so
 * an ordered comparison falls through to a TypeError. */
PyRtObject *pyrt_str_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (!pyrt_str_check(b) || (op != Py_EQ && op != Py_NE))
    return &pyrt_NotImplemented;
  PyRtStrObject *x = (PyRtStrObject *)a;
  PyRtStrObject *y = (PyRtStrObject *)b;
  bool equal = x->length == y->length;
  for (int64_t i = 0; equal && i < PYRT_STR_CAPACITY && i < x->length; ++i)
    equal = x->data[i] == y->data[i];
  return pyrt_bool_from(op == Py_EQ ? equal : !equal);
}
