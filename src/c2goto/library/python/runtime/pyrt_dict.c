#include "pyrt.h"

int64_t pyrt_dict_length(PyRtObject *o);
PyRtObject *pyrt_dict_subscript(PyRtObject *o, PyRtObject *key);
void pyrt_dict_ass_subscript(PyRtObject *o, PyRtObject *key, PyRtObject *value);

PyRtSequenceMethods pyrt_dict_as_sequence = {.sq_length = pyrt_dict_length};

PyRtMappingMethods pyrt_dict_as_mapping = {
  .mp_subscript = pyrt_dict_subscript,
  .mp_ass_subscript = pyrt_dict_ass_subscript};

PyRtTypeObject PyRtDict_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "dict",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_dict_as_sequence,
  .tp_as_mapping = &pyrt_dict_as_mapping};

PyRtObject *pyrt_dict_new(void)
{
  PyRtDictObject *d = __ESBMC_alloca(sizeof(PyRtDictObject));
  d->ob_type = &PyRtDict_Type;
  d->size = 0;
  return (PyRtObject *)d;
}

int64_t pyrt_dict_length(PyRtObject *o)
{
  return ((PyRtDictObject *)o)->size;
}

/* Equal keys must hash equally, so an integral float hashes as the integer it
 * equals and 1 still finds 1.0, as in CPython. Types this model cannot hash
 * share one bucket and are separated by pyrt_key_equal. */
int64_t pyrt_key_hash(PyRtObject *key)
{
  if (pyrt_long_check(key))
    return ((PyRtLongObject *)key)->value;
  if (key->ob_type == &PyRtStr_Type)
  {
    PyRtStrObject *s = (PyRtStrObject *)key;
    int64_t hash = s->length;
    #pragma unroll
    for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < s->length; ++i)
      hash = hash * 31 + s->data[i];
    return hash;
  }
  if (pyrt_float_check(key))
  {
    double value = ((PyRtFloatObject *)key)->value;
    int64_t integral = (int64_t)value;
    return value == (double)integral ? integral : 0;
  }
  return 0;
}

bool pyrt_key_equal(PyRtObject *a, PyRtObject *b)
{
  if (a == b)
    return true;
  if (pyrt_long_check(a) && pyrt_long_check(b))
    return ((PyRtLongObject *)a)->value == ((PyRtLongObject *)b)->value;
  return pyrt_is_true(pyrt_richcompare(a, b, Py_EQ));
}

/* The hash comparison is a pair of integer reads, so a lookup only reaches
 * the full comparison for an entry whose hash already matches. */
int64_t pyrt_dict_find(PyRtDictObject *d, PyRtObject *key)
{
  int64_t hash = pyrt_key_hash(key);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < d->size; ++i)
    if (d->hashes[i] == hash && pyrt_key_equal(d->keys[i], key))
      return i;
  return -1;
}

PyRtObject *pyrt_dict_subscript(PyRtObject *o, PyRtObject *key)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  int64_t at = pyrt_dict_find(d, key);
  if (at < 0)
    PYRT_RAISE("KeyError");
  return d->values[at];
}

void pyrt_dict_ass_subscript(PyRtObject *o, PyRtObject *key, PyRtObject *value)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  int64_t at = pyrt_dict_find(d, key);
  if (at >= 0)
  {
    d->values[at] = value;
    return;
  }
  if (d->size >= PYRT_DICT_CAPACITY)
    PYRT_RAISE("pyrt: more dict entries than the model holds");
  d->hashes[d->size] = pyrt_key_hash(key);
  d->keys[d->size] = key;
  d->values[d->size] = value;
  d->size++;
}

PyRtObject *pyrt_dict_key_at(PyRtObject *o, int64_t i)
{
  return ((PyRtDictObject *)o)->keys[i];
}

/* Entries keep insertion order, as CPython's dicts do since 3.7, so removing
 * one shifts the rest down rather than leaving a hole. */
static void pyrt_dict_remove_at(PyRtDictObject *d, int64_t at)
{
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY - 1; ++i)
    if (i >= at && i + 1 < d->size)
    {
      d->hashes[i] = d->hashes[i + 1];
      d->keys[i] = d->keys[i + 1];
      d->values[i] = d->values[i + 1];
    }
  d->size--;
}

/* dflt == 0 means the caller passed no default, which for get() is None and
 * for pop() is a KeyError. */
PyRtObject *pyrt_dictmeth_get(PyRtObject *o, PyRtObject *key, PyRtObject *dflt)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  int64_t at = pyrt_dict_find(d, key);
  if (at >= 0)
    return d->values[at];
  return dflt ? dflt : &pyrt_None;
}

/* A set, not a list: CPython's keys view is set-like, so d.keys() == {1} holds
 * (#7553). Keys are already unique, and this model's sets keep insertion
 * order, so iterating one still yields them in the dict's order. */
PyRtObject *pyrt_dictmeth_keys(PyRtObject *o)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  PyRtObject *result = pyrt_set_new();
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < d->size; ++i)
    pyrt_set_add(result, d->keys[i]);
  return result;
}

PyRtObject *pyrt_dictmeth_values(PyRtObject *o)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  PyRtObject *result = pyrt_list_new();
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < d->size; ++i)
    pyrt_list_append(result, d->values[i]);
  return result;
}

/* A list of pairs rather than a view: iterating and unpacking it behaves the
 * same, and nothing here depends on a view tracking later writes. */
PyRtObject *pyrt_dictmeth_items(PyRtObject *o)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  PyRtObject *result = pyrt_list_new();
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < d->size; ++i)
  {
    PyRtObject *pair = pyrt_tuple_new();
    pyrt_tuple_append(pair, d->keys[i]);
    pyrt_tuple_append(pair, d->values[i]);
    pyrt_list_append(result, pair);
  }
  return result;
}

PyRtObject *pyrt_dictmeth_pop(PyRtObject *o, PyRtObject *key, PyRtObject *dflt)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  int64_t at = pyrt_dict_find(d, key);
  if (at < 0)
  {
    if (dflt)
      return dflt;
    PYRT_RAISE("KeyError");
    return &pyrt_None;
  }
  PyRtObject *value = d->values[at];
  pyrt_dict_remove_at(d, at);
  return value;
}

PyRtObject *
pyrt_dictmeth_setdefault(PyRtObject *o, PyRtObject *key, PyRtObject *dflt)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  int64_t at = pyrt_dict_find(d, key);
  if (at >= 0)
    return d->values[at];
  PyRtObject *value = dflt ? dflt : &pyrt_None;
  pyrt_dict_ass_subscript(o, key, value);
  return value;
}

void pyrt_dictmeth_update(PyRtObject *o, PyRtObject *other)
{
  if (other->ob_type != &PyRtDict_Type)
    PYRT_RAISE("TypeError: update() argument must be a dict");
  PyRtDictObject *s = (PyRtDictObject *)other;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < s->size; ++i)
    pyrt_dict_ass_subscript(o, s->keys[i], s->values[i]);
}

void pyrt_dictmeth_clear(PyRtObject *o)
{
  ((PyRtDictObject *)o)->size = 0;
}

PyRtObject *pyrt_dictmeth_copy(PyRtObject *o)
{
  PyRtDictObject *d = (PyRtDictObject *)o;
  PyRtObject *result = pyrt_dict_new();
  #pragma unroll
  for (int64_t i = 0; i < PYRT_DICT_CAPACITY && i < d->size; ++i)
    pyrt_dict_ass_subscript(result, d->keys[i], d->values[i]);
  return result;
}
