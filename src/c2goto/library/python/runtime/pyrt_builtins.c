#include "pyrt.h"

PyRtObject *pyrt_builtin_abs(PyRtObject *o)
{
  if (pyrt_long_check(o))
  {
    int64_t value = ((PyRtLongObject *)o)->value;
    return value < 0 ? pyrt_long_from(-value) : o;
  }
  if (pyrt_float_check(o))
  {
    double value = ((PyRtFloatObject *)o)->value;
    return value < 0.0 ? pyrt_float_from(-value) : o;
  }
  PYRT_RAISE("TypeError: bad operand type for abs()");
  return &pyrt_None;
}

PyRtObject *pyrt_builtin_all(PyRtObject *o)
{
  int64_t length = pyrt_iter_length(o);
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < length; ++i)
    if (!pyrt_is_true(pyrt_iter_item(o, i)))
      return pyrt_bool_from(false);
  return pyrt_bool_from(true);
}

PyRtObject *pyrt_builtin_any(PyRtObject *o)
{
  int64_t length = pyrt_iter_length(o);
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < length; ++i)
    if (pyrt_is_true(pyrt_iter_item(o, i)))
      return pyrt_bool_from(true);
  return pyrt_bool_from(false);
}

PyRtObject *pyrt_builtin_sum(PyRtObject *o)
{
  int64_t length = pyrt_iter_length(o);
  PyRtObject *total = pyrt_long_from(0);
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < length; ++i)
    total = pyrt_number_add(total, pyrt_iter_item(o, i));
  return total;
}

/* sum(iterable, start) begins the fold at `start` rather than 0. */
PyRtObject *pyrt_builtin_sum_start(PyRtObject *o, PyRtObject *start)
{
  int64_t length = pyrt_iter_length(o);
  PyRtObject *total = start;
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < length; ++i)
    total = pyrt_number_add(total, pyrt_iter_item(o, i));
  return total;
}

PyRtObject *pyrt_builtin_min2(PyRtObject *a, PyRtObject *b)
{
  return pyrt_is_true(pyrt_richcompare(b, a, Py_LT)) ? b : a;
}

PyRtObject *pyrt_builtin_max2(PyRtObject *a, PyRtObject *b)
{
  return pyrt_is_true(pyrt_richcompare(b, a, Py_GT)) ? b : a;
}

PyRtObject *pyrt_builtin_min_iter(PyRtObject *o)
{
  int64_t length = pyrt_iter_length(o);
  if (length == 0)
  {
    PYRT_RAISE("ValueError: min() arg is an empty sequence");
    return &pyrt_None;
  }
  PyRtObject *best = pyrt_iter_item(o, 0);
  for (int64_t i = 1; i < PYRT_LIST_CAPACITY && i < length; ++i)
    best = pyrt_builtin_min2(best, pyrt_iter_item(o, i));
  return best;
}

PyRtObject *pyrt_builtin_max_iter(PyRtObject *o)
{
  int64_t length = pyrt_iter_length(o);
  if (length == 0)
  {
    PYRT_RAISE("ValueError: max() arg is an empty sequence");
    return &pyrt_None;
  }
  PyRtObject *best = pyrt_iter_item(o, 0);
  for (int64_t i = 1; i < PYRT_LIST_CAPACITY && i < length; ++i)
    best = pyrt_builtin_max2(best, pyrt_iter_item(o, i));
  return best;
}
