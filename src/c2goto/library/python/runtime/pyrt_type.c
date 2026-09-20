#include "pyrt.h"

PyRtNumberMethods pyrt_object_as_number;
PyRtSequenceMethods pyrt_object_as_sequence;
PyRtMappingMethods pyrt_object_as_mapping;

PyRtTypeObject PyRtObject_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "object",
  .tp_as_number = &pyrt_object_as_number,
  .tp_as_sequence = &pyrt_object_as_sequence,
  .tp_as_mapping = &pyrt_object_as_mapping};

PyRtObject *pyrt_attrs_find(PyRtAttrs *attrs, const char *name)
{
  if (!attrs)
    return 0;
  /* The capacity keeps the loop finite once the size is only known as a
   * merged symbol. */
  #pragma unroll
  for (int64_t i = 0; i < PYRT_ATTRS_CAPACITY && i < attrs->size; ++i)
    if (attrs->names[i] == name)
      return attrs->values[i];
  return 0;
}

void pyrt_attrs_set(PyRtAttrs *attrs, const char *name, PyRtObject *value)
{
  #pragma unroll
  for (int64_t i = 0; i < PYRT_ATTRS_CAPACITY && i < attrs->size; ++i)
    if (attrs->names[i] == name)
    {
      attrs->values[i] = value;
      return;
    }
  __ESBMC_assert(
    attrs->size < PYRT_ATTRS_CAPACITY,
    "pyrt: more attributes than the model holds");
  attrs->names[attrs->size] = name;
  attrs->values[attrs->size] = value;
  attrs->size++;
}

/* Single inheritance, so the MRO is the tp_base chain. */
PyRtObject *pyrt_type_lookup(PyRtTypeObject *t, const char *name)
{
  #pragma unroll
  for (int64_t depth = 0; t && depth < PYRT_MAX_CLASSES; t = t->tp_base, ++depth)
  {
    PyRtObject *value = pyrt_attrs_find(t->tp_attrs, name);
    if (value)
      return value;
  }
  __ESBMC_assert(!t, "pyrt: class chain exceeds the model bound");
  return 0;
}

PyRtObject *pyrt_slot_binary(
  PyRtObject *a,
  PyRtObject *b,
  const char *name,
  const char *reflected)
{
  PyRtObject *method = pyrt_type_lookup(a->ob_type, name);
  if (method)
  {
    PyRtArgs args = {a, b};
    PyRtObject *result = pyrt_call(method, args, 2);
    if (result != &pyrt_NotImplemented)
      return result;
  }
  if (a->ob_type != b->ob_type)
  {
    method = pyrt_type_lookup(b->ob_type, reflected);
    if (method)
    {
      PyRtArgs args = {b, a};
      return pyrt_call(method, args, 2);
    }
  }
  return &pyrt_NotImplemented;
}

PyRtObject *pyrt_slot_nb_add(PyRtObject *a, PyRtObject *b)
{
  return pyrt_slot_binary(a, b, pyrt_str___add__, pyrt_str___radd__);
}

PyRtObject *pyrt_slot_nb_subtract(PyRtObject *a, PyRtObject *b)
{
  return pyrt_slot_binary(a, b, pyrt_str___sub__, pyrt_str___rsub__);
}

PyRtObject *pyrt_slot_nb_multiply(PyRtObject *a, PyRtObject *b)
{
  return pyrt_slot_binary(a, b, pyrt_str___mul__, pyrt_str___rmul__);
}

PyRtObject *pyrt_call_special(PyRtArgs args, int64_t nargs, const char *name)
{
  PyRtObject *method = pyrt_type_lookup(args.a0->ob_type, name);
  if (!method)
    PYRT_RAISE("TypeError: special method is not defined");
  return pyrt_call(method, args, nargs);
}

PyRtObject *pyrt_slot_nb_negative(PyRtObject *o)
{
  PyRtArgs args = {o};
  return pyrt_call_special(args, 1, pyrt_str___neg__);
}

bool pyrt_slot_nb_bool(PyRtObject *o)
{
  PyRtArgs args = {o};
  PyRtObject *result = pyrt_call_special(args, 1, pyrt_str___bool__);
  if (
    result != (PyRtObject *)&pyrt_True && result != (PyRtObject *)&pyrt_False)
    PYRT_RAISE("TypeError: __bool__ should return bool");
  return result == (PyRtObject *)&pyrt_True;
}

int64_t pyrt_slot_sq_length(PyRtObject *o)
{
  PyRtArgs args = {o};
  PyRtObject *result = pyrt_call_special(args, 1, pyrt_str___len__);
  if (!pyrt_long_check(result))
    PYRT_RAISE("TypeError: __len__ should return an integer");
  int64_t length = ((PyRtLongObject *)result)->value;
  if (length < 0)
    PYRT_RAISE("ValueError: __len__() should return >= 0");
  return length;
}

PyRtObject *pyrt_slot_mp_subscript(PyRtObject *o, PyRtObject *key)
{
  PyRtArgs args = {o, key};
  return pyrt_call_special(args, 2, pyrt_str___getitem__);
}

void pyrt_slot_mp_ass_subscript(
  PyRtObject *o,
  PyRtObject *key,
  PyRtObject *value)
{
  PyRtArgs args = {o, key, value};
  pyrt_call_special(args, 3, pyrt_str___setitem__);
}

PyRtObject *pyrt_slot_tp_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  static const char *const names[] = {
    pyrt_str___lt__,
    pyrt_str___le__,
    pyrt_str___eq__,
    pyrt_str___ne__,
    pyrt_str___gt__,
    pyrt_str___ge__};
  PyRtObject *method = pyrt_type_lookup(a->ob_type, names[op]);
  if (!method)
    return &pyrt_NotImplemented;
  PyRtArgs args = {a, b};
  return pyrt_call(method, args, 2);
}

bool pyrt_is_slot_name(const char *name)
{
  return name == pyrt_str___add__ || name == pyrt_str___radd__ ||
         name == pyrt_str___sub__ || name == pyrt_str___rsub__ ||
         name == pyrt_str___mul__ || name == pyrt_str___rmul__ ||
         name == pyrt_str___neg__ || name == pyrt_str___bool__ ||
         name == pyrt_str___len__ || name == pyrt_str___getitem__ ||
         name == pyrt_str___setitem__ || name == pyrt_str___lt__ ||
         name == pyrt_str___le__ || name == pyrt_str___eq__ ||
         name == pyrt_str___ne__ || name == pyrt_str___gt__ ||
         name == pyrt_str___ge__;
}

void pyrt_install_slot(PyRtTypeObject *t, const char *name)
{
  if (name == pyrt_str___add__ || name == pyrt_str___radd__)
    t->tp_as_number->nb_add = pyrt_slot_nb_add;
  else if (name == pyrt_str___sub__ || name == pyrt_str___rsub__)
    t->tp_as_number->nb_subtract = pyrt_slot_nb_subtract;
  else if (name == pyrt_str___mul__ || name == pyrt_str___rmul__)
    t->tp_as_number->nb_multiply = pyrt_slot_nb_multiply;
  else if (name == pyrt_str___neg__)
    t->tp_as_number->nb_negative = pyrt_slot_nb_negative;
  else if (name == pyrt_str___bool__)
    t->tp_as_number->nb_bool = pyrt_slot_nb_bool;
  else if (name == pyrt_str___len__)
    t->tp_as_sequence->sq_length = pyrt_slot_sq_length;
  else if (name == pyrt_str___getitem__)
    t->tp_as_mapping->mp_subscript = pyrt_slot_mp_subscript;
  else if (name == pyrt_str___setitem__)
    t->tp_as_mapping->mp_ass_subscript = pyrt_slot_mp_ass_subscript;
  else
    t->tp_richcompare = pyrt_slot_tp_richcompare;
}

void pyrt_inherit_tables(PyRtTypeObject *t)
{
  t->tp_as_number = t->tp_base->tp_as_number;
  t->tp_as_sequence = t->tp_base->tp_as_sequence;
  t->tp_as_mapping = t->tp_base->tp_as_mapping;
  t->tp_richcompare = t->tp_base->tp_richcompare;
}

void pyrt_type_ready(PyRtTypeObject *t)
{
  pyrt_inherit_tables(t);
  t->tp_flags |= PYRT_TPFLAGS_READY;
}

/* Copy-on-write: a type shares its base's slot tables until a special method
 * is set on it. */
void pyrt_own_tables(PyRtTypeObject *t)
{
  if (t->tp_flags & PYRT_TPFLAGS_OWNS_TABLES)
    return;
  PyRtNumberMethods *nb = __ESBMC_alloca(sizeof(PyRtNumberMethods));
  PyRtSequenceMethods *sq = __ESBMC_alloca(sizeof(PyRtSequenceMethods));
  PyRtMappingMethods *mp = __ESBMC_alloca(sizeof(PyRtMappingMethods));
  *nb = *t->tp_as_number;
  *sq = *t->tp_as_sequence;
  *mp = *t->tp_as_mapping;
  t->tp_as_number = nb;
  t->tp_as_sequence = sq;
  t->tp_as_mapping = mp;
  t->tp_flags |= PYRT_TPFLAGS_OWNS_TABLES;
}

/* CPython's update_slot (Objects/typeobject.c): the write reaches every
 * subclass that does not define the name itself. A subclass still sharing
 * tables takes its base's again; one with tables of its own gets the slot
 * written into them. */
void pyrt_update_slot(PyRtTypeObject *t, const char *name)
{
  if (!pyrt_is_slot_name(name))
    return;
  pyrt_own_tables(t);
  pyrt_install_slot(t, name);

  PyRtTypeObject *work[PYRT_MAX_CLASSES] = {0};
  int64_t pending = 0;
  work[pending++] = t;
  #pragma unroll
  for (int64_t visited = 0; pending > 0 && visited < PYRT_MAX_CLASSES;
       ++visited)
  {
    PyRtTypeObject *s = work[--pending];
    int64_t children = 0;
    #pragma unroll
    for (PyRtTypeObject *sub = s->tp_subclass;
         sub && children < PYRT_MAX_CLASSES;
         sub = sub->tp_sibling, ++children)
    {
      if (
        !(sub->tp_flags & PYRT_TPFLAGS_READY) ||
        pyrt_attrs_find(sub->tp_attrs, name))
        continue;
      if (sub->tp_flags & PYRT_TPFLAGS_OWNS_TABLES)
        pyrt_install_slot(sub, name);
      else
        pyrt_inherit_tables(sub);
      if (pending == PYRT_MAX_CLASSES)
        PYRT_RAISE("pyrt: class hierarchy exceeds the model bound");
      work[pending++] = sub;
    }
  }
  __ESBMC_assert(pending == 0, "pyrt: class hierarchy exceeds the model bound");
}
