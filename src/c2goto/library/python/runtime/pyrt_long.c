#include "pyrt.h"

PyRtObject *pyrt_long_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_true_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_floor_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_remainder(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_power(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_negative(PyRtObject *o);
bool pyrt_long_bool(PyRtObject *o);
PyRtObject *pyrt_long_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtNumberMethods pyrt_long_as_number = {
  .nb_add = pyrt_long_add,
  .nb_subtract = pyrt_long_subtract,
  .nb_multiply = pyrt_long_multiply,
  .nb_true_divide = pyrt_long_true_divide,
  .nb_floor_divide = pyrt_long_floor_divide,
  .nb_remainder = pyrt_long_remainder,
  .nb_power = pyrt_long_power,
  .nb_negative = pyrt_long_negative,
  .nb_bool = pyrt_long_bool};

PyRtTypeObject PyRtLong_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "int",
  .tp_base = &PyRtObject_Type,
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

PyRtObject *pyrt_nondet_bool(void)
{
  return pyrt_bool_from(nondet_bool());
}

PyRtObject *pyrt_nondet_int(void)
{
  return pyrt_long_from(nondet_long());
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

/* `/` is true division: int / int is a float, as in Python 3. */
PyRtObject *pyrt_long_true_divide(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  int64_t y = ((PyRtLongObject *)b)->value;
  if (y == 0)
  {
    PYRT_RAISE("ZeroDivisionError: division by zero");
    return &pyrt_NotImplemented;
  }
  return pyrt_float_from((double)((PyRtLongObject *)a)->value / (double)y);
}

/* Python floors toward negative infinity; C truncates toward zero. */
PyRtObject *pyrt_long_floor_divide(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  int64_t x = ((PyRtLongObject *)a)->value;
  int64_t y = ((PyRtLongObject *)b)->value;
  if (y == 0)
  {
    PYRT_RAISE("ZeroDivisionError: integer division or modulo by zero");
    return &pyrt_NotImplemented;
  }
  int64_t q = x / y;
  if (x % y != 0 && (x < 0) != (y < 0))
    q--;
  return pyrt_long_from(q);
}

/* The remainder takes the divisor's sign, so -7 % 3 is 2. */
PyRtObject *pyrt_long_remainder(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  int64_t x = ((PyRtLongObject *)a)->value;
  int64_t y = ((PyRtLongObject *)b)->value;
  if (y == 0)
  {
    PYRT_RAISE("ZeroDivisionError: integer division or modulo by zero");
    return &pyrt_NotImplemented;
  }
  int64_t r = x % y;
  if (r != 0 && (r < 0) != (y < 0))
    r += y;
  return pyrt_long_from(r);
}

PyRtObject *pyrt_long_power(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_long_check(a) || !pyrt_long_check(b))
    return &pyrt_NotImplemented;
  int64_t base = ((PyRtLongObject *)a)->value;
  int64_t e = ((PyRtLongObject *)b)->value;
  if (e < 0)
  {
    PYRT_RAISE("pyrt: negative exponent is not modelled");
    return &pyrt_NotImplemented;
  }
  if (e > PYRT_POW_BOUND)
  {
    PYRT_RAISE("pyrt: exponent exceeds the modelled bound");
    return &pyrt_NotImplemented;
  }
  int64_t result = 1;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_POW_BOUND && i < e; ++i)
    result *= base;
  return pyrt_long_from(result);
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
