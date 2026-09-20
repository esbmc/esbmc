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
  PyRtInstanceObject *o = __ESBMC_alloca(sizeof(PyRtInstanceObject));
  o->ob_type = t;
  o->attrs.size = 0;
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
      pyrt_attrs_find(&((PyRtInstanceObject *)o)->attrs, name);
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
  pyrt_attrs_set(&((PyRtInstanceObject *)o)->attrs, name, value);
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
  if (o->ob_type == &PyRtSet_Type)
  {
    if (name == pyrt_str_add || name == pyrt_str_discard ||
        name == pyrt_str_remove)
    {
      if (nargs != 1)
        PYRT_RAISE("TypeError: this set method takes exactly one argument");
      if (name == pyrt_str_add)
        pyrt_set_add(o, args.a0);
      else if (name == pyrt_str_discard)
        pyrt_set_discard(o, args.a0);
      else
        pyrt_set_remove(o, args.a0);
      return &pyrt_None;
    }
  }
  if (o->ob_type == &PyRtStr_Type)
  {
    if (name == pyrt_str_split)
    {
      if (nargs > 1)
        PYRT_RAISE("TypeError: split() takes at most one argument");
      return pyrt_strmeth_split(o, nargs == 1 ? args.a0 : 0);
    }
    if (name == pyrt_str_join)
    {
      if (nargs != 1)
        PYRT_RAISE("TypeError: join() takes exactly one argument");
      return pyrt_strmeth_join(o, args.a0);
    }
    if (name == pyrt_str_replace)
    {
      if (nargs != 2)
        PYRT_RAISE("TypeError: replace() takes exactly two arguments");
      return pyrt_strmeth_replace(o, args.a0, args.a1);
    }
    if (
      name == pyrt_str_strip || name == pyrt_str_lstrip ||
      name == pyrt_str_rstrip)
    {
      if (nargs > 1)
        PYRT_RAISE("TypeError: this strip method takes at most one argument");
      return pyrt_strmeth_strip(
        o,
        nargs == 1 ? args.a0 : 0,
        name != pyrt_str_rstrip,
        name != pyrt_str_lstrip);
    }
  }
  if (o->ob_type == &PyRtDict_Type)
  {
    if (name == pyrt_str_get || name == pyrt_str_pop ||
        name == pyrt_str_setdefault)
    {
      if (nargs < 1 || nargs > 2)
        PYRT_RAISE("TypeError: this dict method takes one or two arguments");
      PyRtObject *dflt = nargs == 2 ? args.a1 : 0;
      if (name == pyrt_str_get)
        return pyrt_dictmeth_get(o, args.a0, dflt);
      if (name == pyrt_str_pop)
        return pyrt_dictmeth_pop(o, args.a0, dflt);
      return pyrt_dictmeth_setdefault(o, args.a0, dflt);
    }
    if (name == pyrt_str_keys || name == pyrt_str_values ||
        name == pyrt_str_items || name == pyrt_str_copy ||
        name == pyrt_str_clear)
    {
      if (nargs != 0)
        PYRT_RAISE("TypeError: this dict method takes no arguments");
      if (name == pyrt_str_keys)
        return pyrt_dictmeth_keys(o);
      if (name == pyrt_str_values)
        return pyrt_dictmeth_values(o);
      if (name == pyrt_str_items)
        return pyrt_dictmeth_items(o);
      if (name == pyrt_str_copy)
        return pyrt_dictmeth_copy(o);
      pyrt_dictmeth_clear(o);
      return &pyrt_None;
    }
    if (name == pyrt_str_update)
    {
      if (nargs != 1)
        PYRT_RAISE("TypeError: update() takes exactly one argument");
      pyrt_dictmeth_update(o, args.a0);
      return &pyrt_None;
    }
  }
  if (o->ob_type == &PyRtList_Type)
  {
    if (name == pyrt_str_pop)
    {
      if (nargs > 1)
        PYRT_RAISE("TypeError: pop() takes at most one argument");
      return pyrt_listmeth_pop(o, nargs == 1 ? args.a0 : 0);
    }
    if (name == pyrt_str_extend || name == pyrt_str_remove ||
        name == pyrt_str_index || name == pyrt_str_count)
    {
      if (nargs != 1)
        PYRT_RAISE("TypeError: this list method takes exactly one argument");
      if (name == pyrt_str_extend)
      {
        pyrt_listmeth_extend(o, args.a0);
        return &pyrt_None;
      }
      if (name == pyrt_str_remove)
      {
        pyrt_listmeth_remove(o, args.a0);
        return &pyrt_None;
      }
      if (name == pyrt_str_index)
        return pyrt_listmeth_index(o, args.a0);
      return pyrt_listmeth_count(o, args.a0);
    }
    if (name == pyrt_str_insert)
    {
      if (nargs != 2)
        PYRT_RAISE("TypeError: insert() takes exactly two arguments");
      pyrt_listmeth_insert(o, args.a0, args.a1);
      return &pyrt_None;
    }
    if (name == pyrt_str_sort || name == pyrt_str_clear ||
        name == pyrt_str_copy)
    {
      if (nargs != 0)
        PYRT_RAISE("TypeError: this list method takes no arguments");
      if (name == pyrt_str_copy)
        return pyrt_listmeth_copy(o);
      if (name == pyrt_str_sort)
        pyrt_listmeth_sort(o);
      else
        pyrt_listmeth_clear(o);
      return &pyrt_None;
    }
  }
  return pyrt_call(pyrt_getattr(o, name), args, nargs);
}
