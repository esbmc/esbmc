#include "pyrt.h"

/* Iteration walks indices over the length read once at the top of the loop.
 * Resizing a container while iterating it raises RuntimeError in CPython and
 * is not modelled here. */
int64_t pyrt_iter_length(PyRtObject *o)
{
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (!sq || !sq->sq_length)
    PYRT_RAISE("TypeError: object is not iterable");
  return sq->sq_length(o);
}

PyRtObject *pyrt_iter_item(PyRtObject *o, int64_t i)
{
  if (o->ob_type == &PyRtDict_Type)
    return pyrt_dict_key_at(o, i);
  PyRtSequenceMethods *sq = o->ob_type->tp_as_sequence;
  if (!sq || !sq->sq_item)
    PYRT_RAISE("TypeError: object is not iterable");
  return sq->sq_item(o, i);
}

PyRtObject *pyrt_contains(PyRtObject *container, PyRtObject *item)
{
  if (container->ob_type == &PyRtStr_Type)
  {
    if (container->ob_type != item->ob_type)
      PYRT_RAISE("TypeError: 'in <string>' requires string as left operand");
    return pyrt_bool_from(
      pyrt_str_search(
        (PyRtStrObject *)container, (PyRtStrObject *)item, false) >= 0);
  }
  if (container->ob_type == &PyRtSet_Type)
    return pyrt_bool_from(
      pyrt_set_find((PyRtSetObject *)container, item) >= 0);
  if (container->ob_type == &PyRtDict_Type)
    return pyrt_bool_from(
      pyrt_dict_find((PyRtDictObject *)container, item) >= 0);
  int64_t length = pyrt_iter_length(container);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < length; ++i)
    if (pyrt_key_equal(pyrt_iter_item(container, i), item))
      return pyrt_bool_from(true);
  return pyrt_bool_from(false);
}

int64_t pyrt_as_index(PyRtObject *o)
{
  if (!pyrt_long_check(o))
    PYRT_RAISE("TypeError: expected an integer");
  return ((PyRtLongObject *)o)->value;
}
