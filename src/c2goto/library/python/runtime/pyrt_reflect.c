#include "pyrt.h"

PyRtObject *pyrt_type_of(PyRtObject *o)
{
  return (PyRtObject *)o->ob_type;
}

bool pyrt_isinstance(PyRtObject *o, PyRtObject *cls)
{
  if (cls->ob_type != &PyRtType_Type)
    PYRT_RAISE("TypeError: isinstance() arg 2 must be a type");
  PyRtTypeObject *t = o->ob_type;
  for (int64_t depth = 0; t && depth < PYRT_MAX_CLASSES; t = t->tp_base, ++depth)
    if (t == (PyRtTypeObject *)cls)
      return true;
  return false;
}

bool pyrt_hasattr(PyRtObject *o, const char *name)
{
  if (name == pyrt_str___class__)
    return true;
  if (o->ob_type == &PyRtType_Type)
    return pyrt_type_lookup((PyRtTypeObject *)o, name) != 0;
  if (
    (o->ob_type->tp_flags & PYRT_TPFLAGS_HEAPTYPE) &&
    pyrt_attrs_find(((PyRtInstanceObject *)o)->attrs, name))
    return true;
  return pyrt_type_lookup(o->ob_type, name) != 0;
}
