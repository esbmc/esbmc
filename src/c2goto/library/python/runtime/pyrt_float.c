#include "pyrt.h"

PyRtObject *pyrt_float_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_true_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_floor_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_remainder(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_power(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_float_negative(PyRtObject *o);
bool pyrt_float_bool(PyRtObject *o);
PyRtObject *pyrt_float_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtNumberMethods pyrt_float_as_number = {
  .nb_add = pyrt_float_add,
  .nb_subtract = pyrt_float_subtract,
  .nb_multiply = pyrt_float_multiply,
  .nb_true_divide = pyrt_float_true_divide,
  .nb_floor_divide = pyrt_float_floor_divide,
  .nb_remainder = pyrt_float_remainder,
  .nb_power = pyrt_float_power,
  .nb_negative = pyrt_float_negative,
  .nb_bool = pyrt_float_bool};

PyRtTypeObject PyRtFloat_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "float",
  .tp_base = &PyRtObject_Type,
  .tp_as_number = &pyrt_float_as_number,
  .tp_richcompare = pyrt_float_richcompare};

bool pyrt_float_check(PyRtObject *o)
{
  return o->ob_type == &PyRtFloat_Type;
}

bool pyrt_number_check(PyRtObject *o)
{
  return pyrt_long_check(o) || pyrt_float_check(o);
}

PyRtObject *pyrt_float_from(double v)
{
  PyRtFloatObject *o = __ESBMC_alloca(sizeof(PyRtFloatObject));
  o->ob_type = &PyRtFloat_Type;
  o->value = v;
  return (PyRtObject *)o;
}

/* int and bool widen to double, which is how CPython mixes them with float. */
double pyrt_number_as_double(PyRtObject *o)
{
  if (pyrt_float_check(o))
    return ((PyRtFloatObject *)o)->value;
  return (double)((PyRtLongObject *)o)->value;
}

PyRtObject *pyrt_float_add(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  return pyrt_float_from(pyrt_number_as_double(a) + pyrt_number_as_double(b));
}

PyRtObject *pyrt_float_subtract(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  return pyrt_float_from(pyrt_number_as_double(a) - pyrt_number_as_double(b));
}

PyRtObject *pyrt_float_multiply(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  return pyrt_float_from(pyrt_number_as_double(a) * pyrt_number_as_double(b));
}

/* floor() without libm: the cast truncates toward zero, so a negative
 * non-integral value is one short. Values beyond int64 are not modelled. */
static double pyrt_floor(double q)
{
  double truncated = (double)(int64_t)q;
  return truncated > q ? truncated - 1.0 : truncated;
}

PyRtObject *pyrt_float_true_divide(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  double y = pyrt_number_as_double(b);
  if (y == 0.0)
  {
    PYRT_RAISE("ZeroDivisionError: float division by zero");
    return &pyrt_NotImplemented;
  }
  return pyrt_float_from(pyrt_number_as_double(a) / y);
}

PyRtObject *pyrt_float_floor_divide(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  double y = pyrt_number_as_double(b);
  if (y == 0.0)
  {
    PYRT_RAISE("ZeroDivisionError: float floor division by zero");
    return &pyrt_NotImplemented;
  }
  return pyrt_float_from(pyrt_floor(pyrt_number_as_double(a) / y));
}

PyRtObject *pyrt_float_remainder(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  double x = pyrt_number_as_double(a);
  double y = pyrt_number_as_double(b);
  if (y == 0.0)
  {
    PYRT_RAISE("ZeroDivisionError: float modulo");
    return &pyrt_NotImplemented;
  }
  return pyrt_float_from(x - pyrt_floor(x / y) * y);
}

PyRtObject *pyrt_float_power(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  double base = pyrt_number_as_double(a);
  double e = pyrt_number_as_double(b);
  int64_t n = (int64_t)e;
  if ((double)n != e)
  {
    PYRT_RAISE("pyrt: non-integral exponent is not modelled");
    return &pyrt_NotImplemented;
  }
  int64_t magnitude = n < 0 ? -n : n;
  if (magnitude > PYRT_POW_BOUND)
  {
    PYRT_RAISE("pyrt: exponent exceeds the modelled bound");
    return &pyrt_NotImplemented;
  }
  double result = 1.0;
  for (int64_t i = 0; i < PYRT_POW_BOUND && i < magnitude; ++i)
    result *= base;
  if (n < 0)
  {
    if (result == 0.0)
    {
      PYRT_RAISE("ZeroDivisionError: 0.0 cannot be raised to a negative power");
      return &pyrt_NotImplemented;
    }
    result = 1.0 / result;
  }
  return pyrt_float_from(result);
}

PyRtObject *pyrt_float_negative(PyRtObject *o)
{
  return pyrt_float_from(-((PyRtFloatObject *)o)->value);
}

bool pyrt_float_bool(PyRtObject *o)
{
  return ((PyRtFloatObject *)o)->value != 0.0;
}

PyRtObject *pyrt_float_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (!pyrt_number_check(a) || !pyrt_number_check(b))
    return &pyrt_NotImplemented;
  double x = pyrt_number_as_double(a);
  double y = pyrt_number_as_double(b);
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
