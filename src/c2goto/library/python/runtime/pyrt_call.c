#include "pyrt.h"

PyRtTypeObject PyRtFunction_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "function",
  .tp_base = &PyRtObject_Type};

PyRtTypeObject PyRtMethod_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "method",
  .tp_base = &PyRtObject_Type};

PyRtObject *pyrt_method_new(PyRtObject *self, PyRtObject *function)
{
  PyRtMethodObject *m = __ESBMC_alloca(sizeof(PyRtMethodObject));
  m->ob_type = &PyRtMethod_Type;
  m->self = self;
  m->function = function;
  return (PyRtObject *)m;
}

PyRtObject *
pyrt_call_function(PyRtObject *callable, PyRtArgs args, int64_t nargs)
{
  PyRtFunctionObject *f = (PyRtFunctionObject *)callable;
  if (f->arity != nargs)
    PYRT_RAISE("TypeError: function called with the wrong number of arguments");
  return f->code(args);
}

PyRtArgs pyrt_prepend(PyRtObject *first, PyRtArgs args, int64_t nargs)
{
  if (nargs >= PYRT_MAX_ARGS)
    PYRT_RAISE("pyrt: more arguments than the model passes");
  PyRtArgs all = {first, args.a0, args.a1, args.a2, args.a3, args.a4};
  return all;
}

PyRtObject *pyrt_type_call(PyRtTypeObject *t, PyRtArgs args, int64_t nargs)
{
  PyRtAttrs *attrs = __ESBMC_alloca(sizeof(PyRtAttrs));
  attrs->size = 0;
  PyRtInstanceObject *o = __ESBMC_alloca(sizeof(PyRtInstanceObject));
  o->ob_type = t;
  o->attrs = attrs;
  PyRtObject *self = (PyRtObject *)o;

  PyRtObject *init = pyrt_type_lookup(t, pyrt_str___init__);
  if (!init)
  {
    if (nargs != 0)
      PYRT_RAISE("TypeError: this constructor takes no arguments");
    return self;
  }
  if (init->ob_type != &PyRtFunction_Type)
    PYRT_RAISE("TypeError: __init__ is not a function");
  if (
    pyrt_call_function(init, pyrt_prepend(self, args, nargs), nargs + 1) !=
    &pyrt_None)
    PYRT_RAISE("TypeError: __init__() should return None");
  return self;
}

PyRtObject *pyrt_call(PyRtObject *callable, PyRtArgs args, int64_t nargs)
{
  PyRtTypeObject *t = callable->ob_type;
  if (t == &PyRtFunction_Type)
    return pyrt_call_function(callable, args, nargs);
  if (t == &PyRtMethod_Type)
  {
    PyRtMethodObject *m = (PyRtMethodObject *)callable;
    return pyrt_call_function(
      m->function, pyrt_prepend(m->self, args, nargs), nargs + 1);
  }
  if (
    t == &PyRtType_Type &&
    (((PyRtTypeObject *)callable)->tp_flags & PYRT_TPFLAGS_HEAPTYPE))
    return pyrt_type_call((PyRtTypeObject *)callable, args, nargs);
  PYRT_RAISE("TypeError: object is not callable");
  return &pyrt_None;
}

/* CPython's generic getattr, minus descriptors other than functions: the
 * instance's own attributes shadow the class's. */
PyRtObject *pyrt_getattr(PyRtObject *o, const char *name)
{
  if (name == pyrt_str___class__)
    return (PyRtObject *)o->ob_type;
  if (o->ob_type == &PyRtType_Type)
  {
    PyRtObject *value = pyrt_type_lookup((PyRtTypeObject *)o, name);
    if (!value)
      PYRT_RAISE("AttributeError: type object has no such attribute");
    return value;
  }
  if (o->ob_type->tp_flags & PYRT_TPFLAGS_HEAPTYPE)
  {
    PyRtObject *value =
      pyrt_attrs_find(((PyRtInstanceObject *)o)->attrs, name);
    if (value)
      return value;
  }
  PyRtObject *value = pyrt_type_lookup(o->ob_type, name);
  if (!value)
    PYRT_RAISE("AttributeError: object has no such attribute");
  if (value->ob_type == &PyRtFunction_Type)
    return pyrt_method_new(o, value);
  return value;
}

void pyrt_setattr(PyRtObject *o, const char *name, PyRtObject *value)
{
  if (o->ob_type == &PyRtType_Type)
  {
    PyRtTypeObject *t = (PyRtTypeObject *)o;
    if (!(t->tp_flags & PYRT_TPFLAGS_HEAPTYPE))
      PYRT_RAISE("TypeError: cannot set attributes of a built-in type");
    pyrt_attrs_set(t->tp_attrs, name, value);
    pyrt_update_slot(t, name);
    return;
  }
  if (!(o->ob_type->tp_flags & PYRT_TPFLAGS_HEAPTYPE))
    PYRT_RAISE("AttributeError: object has no attribute __dict__");
  pyrt_attrs_set(((PyRtInstanceObject *)o)->attrs, name, value);
}

PyRtObject *
pyrt_call_method(PyRtObject *o, const char *name, PyRtArgs args, int64_t nargs)
{
  if (o->ob_type == &PyRtList_Type && name == pyrt_str_append)
  {
    if (nargs != 1)
      PYRT_RAISE("TypeError: append() takes exactly one argument");
    pyrt_list_append(o, args.a0);
    return &pyrt_None;
  }
  return pyrt_call(pyrt_getattr(o, name), args, nargs);
}
