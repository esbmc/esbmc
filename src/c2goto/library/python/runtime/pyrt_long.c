#include "pyrt.h"

PyRtObject *pyrt_long_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_negative(PyRtObject *o);
bool pyrt_long_bool(PyRtObject *o);
PyRtObject *pyrt_long_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtNumberMethods pyrt_long_as_number = {
  .nb_add = pyrt_long_add,
  .nb_subtract = pyrt_long_subtract,
  .nb_multiply = pyrt_long_multiply,
  .nb_negative = pyrt_long_negative,
  .nb_bool = pyrt_long_bool};

PyRtTypeObject PyRtLong_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "int",
  .tp_as_number = &pyrt_long_as_number,
  .tp_richcompare = pyrt_long_richcompare};

PyRtTypeObject PyRtBool_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "bool",
  .tp_base = &PyRtLong_Type,
  .tp_as_number = &pyrt_long_as_number,
  .tp_richcompare = pyrt_long_richcompare};

PyRtLongObject pyrt_True = {&PyRtBool_Type, 1};
PyRtLongObject pyrt_False = {&PyRtBool_Type, 0};

PyRtObject *pyrt_bool_from(bool b)
{
  return b ? (PyRtObject *)&pyrt_True : (PyRtObject *)&pyrt_False;
}

bool pyrt_long_check(PyRtObject *o)
{
  return o->ob_type == &PyRtLong_Type || o->ob_type == &PyRtBool_Type;
}

PyRtObject *pyrt_long_from(int64_t v)
{
  PyRtLongObject *o = __ESBMC_alloca(sizeof(PyRtLongObject));
  o->ob_type = &PyRtLong_Type;
  o->value = v;
  return (PyRtObject *)o;
}

PyRtObject *pyrt_long_add(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  return pyrt_long_from(
    ((PyRtLongObject *)a)->value + ((PyRtLongObject *)b)->value);
}

PyRtObject *pyrt_long_subtract(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  return pyrt_long_from(
    ((PyRtLongObject *)a)->value - ((PyRtLongObject *)b)->value);
}

PyRtObject *pyrt_long_multiply(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  return pyrt_long_from(
    ((PyRtLongObject *)a)->value * ((PyRtLongObject *)b)->value);
}

PyRtObject *pyrt_long_negative(PyRtObject *o)
{
  return pyrt_long_from(-((PyRtLongObject *)o)->value);
}

bool pyrt_long_bool(PyRtObject *o)
{
  return ((PyRtLongObject *)o)->value != 0;
}

PyRtObject *pyrt_long_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  int64_t x = ((PyRtLongObject *)a)->value;
  int64_t y = ((PyRtLongObject *)b)->value;
  switch (op)
  {
  case Py_LT:
    return pyrt_bool_from(x < y);
  case Py_LE:
    return pyrt_bool_from(x <= y);
  case Py_EQ:
    return pyrt_bool_from(x == y);
  case Py_NE:
    return pyrt_bool_from(x != y);
  case Py_GT:
    return pyrt_bool_from(x > y);
  default:
    return pyrt_bool_from(x >= y);
  }
}
