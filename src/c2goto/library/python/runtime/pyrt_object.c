#include "pyrt.h"

PyRtTypeObject PyRtType_Type = {&PyRtType_Type, "type"};
PyRtTypeObject PyRtNone_Type = {&PyRtType_Type, "NoneType"};
PyRtTypeObject PyRtNotImplemented_Type = {&PyRtType_Type, "NotImplementedType"};

PyRtObject pyrt_None = {&PyRtNone_Type};
PyRtObject pyrt_NotImplemented = {&PyRtNotImplemented_Type};

bool pyrt_is_true(PyRtObject *o)
{
  if (o == &pyrt_None)
    return false;
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
  PYRT_MULTIPLY
} pyrt_binop;

binaryfunc pyrt_number_slot(PyRtTypeObject *t, pyrt_binop op)
{
  PyRtNumberMethods *nb = t->tp_as_number;
  if (!nb)
    return 0;
  if (op == PYRT_ADD)
    return nb->nb_add;
  if (op == PYRT_SUBTRACT)
    return nb->nb_subtract;
  return nb->nb_multiply;
}

/* CPython's binary_op1 (Objects/abstract.c), without the subclass-first rule:
 * the only subclass here is bool, which shares int's slots. */
PyRtObject *pyrt_binary_op1(PyRtObject *a, PyRtObject *b, pyrt_binop op)
{
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
