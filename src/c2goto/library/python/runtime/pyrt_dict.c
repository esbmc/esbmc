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
