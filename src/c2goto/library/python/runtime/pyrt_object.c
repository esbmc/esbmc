#include "pyrt.h"

PyRtTypeObject PyRtType_Type = {&PyRtType_Type, "type"};
PyRtTypeObject PyRtNone_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "NoneType",
  .tp_base = &PyRtObject_Type};
PyRtTypeObject PyRtNotImplemented_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "NotImplementedType",
  .tp_base = &PyRtObject_Type};

PyRtObject pyrt_None = {&PyRtNone_Type};
PyRtObject pyrt_NotImplemented = {&PyRtNotImplemented_Type};

bool pyrt_is_true(PyRtObject *o)
{
  if (o == &pyrt_None)
    return false;
  /* A built-in type refuses slot assignment (pyrt_setattr), so int and bool
   * truthiness is fixed for the whole program and can be read directly. Each
   * slot read this skips carries its own dereference claims, and where the
   * type is known the guard folds away entirely. */
  if (pyrt_long_check(o))
    return ((PyRtLongObject *)o)->value != 0;
  PyRtNumberMethods *nb = o->ob_type->tp_as_number;
  if (nb && nb->nb_bool)
    return nb->nb_bool(o);
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (sq && sq->sq_length)
    return sq->sq_length(o) != 0;
  return true;
}

typedef enum
{
  PYRT_ADD,
  PYRT_SUBTRACT,
  PYRT_MULTIPLY,
  PYRT_TRUE_DIVIDE,
  PYRT_FLOOR_DIVIDE,
  PYRT_REMAINDER,
  PYRT_POWER
} pyrt_binop;

binaryfunc pyrt_number_slot(PyRtTypeObject *t, pyrt_binop op)
{
  PyRtNumberMethods *nb = t->tp_as_number;
  if (!nb)
    return 0;
  switch (op)
  {
  case PYRT_ADD:
    return nb->nb_add;
  case PYRT_SUBTRACT:
    return nb->nb_subtract;
  case PYRT_TRUE_DIVIDE:
    return nb->nb_true_divide;
  case PYRT_FLOOR_DIVIDE:
    return nb->nb_floor_divide;
  case PYRT_REMAINDER:
    return nb->nb_remainder;
  case PYRT_POWER:
    return nb->nb_power;
  default:
    return nb->nb_multiply;
  }
}

/* CPython's binary_op1 (Objects/abstract.c), without the subclass-first rule:
 * the only subclass here is bool, which shares int's slots. */
PyRtObject *pyrt_binary_op1(PyRtObject *a, PyRtObject *b, pyrt_binop op)
{
  /* int and bool share one fixed slot table, so two integer operands reach
   * the implementation without a lookup. `op` is a literal at every call
   * site, so this switch folds; where the operand types are known the guard
   * folds too and the slot reads and indirect call disappear. */
  if (pyrt_long_check(a) && pyrt_long_check(b))
    switch (op)
    {
    case PYRT_ADD:
      return pyrt_long_add(a, b);
    case PYRT_SUBTRACT:
      return pyrt_long_subtract(a, b);
    case PYRT_MULTIPLY:
      return pyrt_long_multiply(a, b);
    case PYRT_TRUE_DIVIDE:
      return pyrt_long_true_divide(a, b);
    case PYRT_FLOOR_DIVIDE:
      return pyrt_long_floor_divide(a, b);
    case PYRT_REMAINDER:
      return pyrt_long_remainder(a, b);
    default:
      return pyrt_long_power(a, b);
    }
  binaryfunc slotv = pyrt_number_slot(a->ob_type, op);
  binaryfunc slotw =
    a->ob_type != b->ob_type ? pyrt_number_slot(b->ob_type, op) : 0;
  if (slotw == slotv)
    slotw = 0;
  if (slotv)
  {
    PyRtObject *result = slotv(a, b);
    if (result != &pyrt_NotImplemented)
      return result;
  }
  if (slotw)
    return slotw(a, b);
  return &pyrt_NotImplemented;
}

PyRtObject *pyrt_number_add(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_ADD);
  if (result != &pyrt_NotImplemented)
    return result;
  PyRtSequenceMethods *sq = a->ob_type->tp_as_sequence;
  if (sq && sq->sq_concat)
    return sq->sq_concat(a, b);
  PYRT_RAISE("TypeError: unsupported operand type(s) for +");
  return result;
}

PyRtObject *pyrt_number_subtract(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_SUBTRACT);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for -");
  return result;
}

PyRtObject *pyrt_number_multiply(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_MULTIPLY);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for *");
  return result;
}

PyRtObject *pyrt_number_true_divide(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_TRUE_DIVIDE);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for /");
  return result;
}

PyRtObject *pyrt_number_floor_divide(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_FLOOR_DIVIDE);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for //");
  return result;
}

PyRtObject *pyrt_number_remainder(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_REMAINDER);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for %");
  return result;
}

PyRtObject *pyrt_number_power(PyRtObject *a, PyRtObject *b)
{
  PyRtObject *result = pyrt_binary_op1(a, b, PYRT_POWER);
  if (result == &pyrt_NotImplemented)
    PYRT_RAISE("TypeError: unsupported operand type(s) for **");
  return result;
}

PyRtObject *pyrt_number_negative(PyRtObject *o)
{
  PyRtNumberMethods *nb = o->ob_type->tp_as_number;
  if (!nb || !nb->nb_negative)
    PYRT_RAISE("TypeError: bad operand type for unary -");
  return nb->nb_negative(o);
}

/* CPython's do_richcompare (Objects/object.c), again without the subclass-first
 * rule. */
PyRtObject *pyrt_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  static const int swapped_op[] = {Py_GT, Py_GE, Py_EQ, Py_NE, Py_LT, Py_LE};
  /* Two integers never reach the reflected operand: pyrt_long_richcompare
   * answers every op, and int's slots cannot be reassigned. */
  if (pyrt_long_check(a) && pyrt_long_check(b))
    return pyrt_long_richcompare(a, b, op);
  richcmpfunc f = a->ob_type->tp_richcompare;
  if (f)
  {
    PyRtObject *result = f(a, b, op);
    if (result != &pyrt_NotImplemented)
      return result;
  }
  f = b->ob_type->tp_richcompare;
  if (f)
  {
    PyRtObject *result = f(b, a, swapped_op[op]);
    if (result != &pyrt_NotImplemented)
      return result;
  }
  if (op == Py_EQ)
    return pyrt_bool_from(a == b);
  if (op == Py_NE)
    return pyrt_bool_from(a != b);
  PYRT_RAISE("TypeError: ordering comparison not supported between instances");
  return &pyrt_NotImplemented;
}

PyRtObject *pyrt_builtin_len(PyRtObject *o)
{
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (!sq || !sq->sq_length)
    PYRT_RAISE("TypeError: object has no len()");
  return pyrt_long_from(sq->sq_length(o));
}

int64_t pyrt_sequence_index(PyRtObject *o, PyRtObject *key)
{
  if (!pyrt_long_check(key))
    PYRT_RAISE("TypeError: sequence indices must be integers");
  int64_t i = ((PyRtLongObject *)key)->value;
  if (i < 0)
    i += o->ob_type->tp_as_sequence->sq_length(o);
  return i;
}

PyRtObject *pyrt_getitem(PyRtObject *o, PyRtObject *key)
{
  PyRtMappingMethods *mp = o->ob_type->tp_as_mapping;
  if (mp && mp->mp_subscript)
    return mp->mp_subscript(o, key);
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (!sq || !sq->sq_item)
    PYRT_RAISE("TypeError: object is not subscriptable");
  return sq->sq_item(o, pyrt_sequence_index(o, key));
}

/* CPython's slice normalisation: a negative bound counts from the end, one
 * out of range clamps rather than raising, and what a missing bound means
 * depends on the step's sign -- `a[::-1]` starts at the last element. */
static int64_t
pyrt_slice_bound(int64_t given, int64_t length, int64_t by)
{
  if (given < 0)
    given += length;
  if (by > 0)
    return given < 0 ? 0 : (given > length ? length : given);
  return given < -1 ? -1 : (given > length - 1 ? length - 1 : given);
}

PyRtObject *pyrt_getslice(
  PyRtObject *o,
  PyRtObject *start,
  PyRtObject *stop,
  PyRtObject *step)
{
  const int64_t length = pyrt_iter_length(o);

  int64_t by = 1;
  if (step != &pyrt_None)
  {
    by = pyrt_as_index(step);
    if (by == 0)
      PYRT_RAISE("ValueError: slice step cannot be zero");
  }

  int64_t from = by > 0 ? 0 : length - 1;
  if (start != &pyrt_None)
    from = pyrt_slice_bound(pyrt_as_index(start), length, by);

  int64_t to = by > 0 ? length : -1;
  if (stop != &pyrt_None)
    to = pyrt_slice_bound(pyrt_as_index(stop), length, by);

  if (o->ob_type == &PyRtStr_Type)
  {
    char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY);
    int64_t taken = 0;
    int64_t at = from;
    #pragma unroll
    for (int64_t k = 0; k < PYRT_STR_CAPACITY; ++k, at += by)
    {
      if (by > 0 ? at >= to : at <= to)
        break;
      __ESBMC_assert(
        taken < PYRT_STR_CAPACITY, "pyrt: slice exceeds the model capacity");
      buffer[taken++] = ((PyRtStrObject *)o)->data[at];
    }
    return pyrt_str_new(buffer, taken);
  }

  const bool as_tuple = o->ob_type == &PyRtTuple_Type;
  PyRtObject *result = as_tuple ? pyrt_tuple_new() : pyrt_list_new();
  int64_t at = from;
  #pragma unroll
  for (int64_t k = 0; k < PYRT_LIST_CAPACITY; ++k, at += by)
  {
    if (by > 0 ? at >= to : at <= to)
      break;
    if (as_tuple)
      pyrt_tuple_append(result, pyrt_iter_item(o, at));
    else
      pyrt_list_append(result, pyrt_iter_item(o, at));
  }
  return result;
}

/* int(x). A float truncates toward zero, as CPython does; bool is an int
 * already. Parsing a string is a decimal scan this does not model. */
PyRtObject *pyrt_to_int(PyRtObject *o)
{
  if (pyrt_long_check(o))
    return pyrt_long_from(((PyRtLongObject *)o)->value);
  if (pyrt_float_check(o))
    return pyrt_long_from((int64_t)((PyRtFloatObject *)o)->value);
  if (o->ob_type == &PyRtStr_Type)
    PYRT_RAISE("pyrt: int() of a string is not modelled");
  PYRT_RAISE("TypeError: int() argument must be a number");
  return &pyrt_None;
}

PyRtObject *pyrt_to_float(PyRtObject *o)
{
  if (pyrt_number_check(o))
    return pyrt_float_from(pyrt_number_as_double(o));
  if (o->ob_type == &PyRtStr_Type)
    PYRT_RAISE("pyrt: float() of a string is not modelled");
  PYRT_RAISE("TypeError: float() argument must be a number");
  return &pyrt_None;
}

/* str(x) only where no rendering is needed. Turning a number into digits
 * wants a buffer and a format this does not model, so it is refused rather
 * than answered with something shorter than the truth. */
PyRtObject *pyrt_to_str(PyRtObject *o)
{
  if (o->ob_type == &PyRtStr_Type)
    return o;
  PYRT_RAISE("pyrt: str() of this type is not modelled");
  return &pyrt_None;
}

/* Used where the caller can turn an error into a Python exception -- inside a
 * try. A missing key is recorded rather than asserted, so the caller throws a
 * KeyError the program can catch. Everything else keeps the ordinary path,
 * whose asserts carry the message a failure is reported with. */
PyRtObject *pyrt_getitem_checked(PyRtObject *o, PyRtObject *key)
{
  if (o->ob_type == &PyRtDict_Type)
  {
    PyRtDictObject *d = (PyRtDictObject *)o;
    int64_t at = pyrt_dict_find(d, key);
    if (at < 0)
    {
      pyrt_set_pending(&PyRtKeyError_Type);
      return 0;
    }
    return d->values[at];
  }
  return pyrt_getitem(o, key);
}

void pyrt_setitem(PyRtObject *o, PyRtObject *key, PyRtObject *value)
{
  PyRtMappingMethods *mp = o->ob_type->tp_as_mapping;
  if (mp && mp->mp_ass_subscript)
  {
    mp->mp_ass_subscript(o, key, value);
    return;
  }
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (!sq || !sq->sq_ass_item)
    PYRT_RAISE("TypeError: object does not support item assignment");
  sq->sq_ass_item(o, pyrt_sequence_index(o, key), value);
}
