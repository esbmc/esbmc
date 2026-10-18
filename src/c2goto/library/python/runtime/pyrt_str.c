#include "pyrt.h"

int64_t pyrt_str_length(PyRtObject *o);
PyRtObject *pyrt_str_item(PyRtObject *o, int64_t i);
PyRtObject *pyrt_str_concat(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_str_richcompare(PyRtObject *a, PyRtObject *b, int op);

PyRtSequenceMethods pyrt_str_as_sequence = {
  .sq_length = pyrt_str_length,
  .sq_concat = pyrt_str_concat,
  .sq_item = pyrt_str_item};

PyRtTypeObject PyRtStr_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "str",
  .tp_base = &PyRtObject_Type,
  .tp_as_sequence = &pyrt_str_as_sequence,
  .tp_richcompare = pyrt_str_richcompare};

bool pyrt_str_check(PyRtObject *o)
{
  return o->ob_type == &PyRtStr_Type;
}

PyRtObject *pyrt_str_new(const char *data, int64_t length)
{
  PyRtStrObject *s = __ESBMC_alloca(sizeof(PyRtStrObject));
  s->ob_type = &PyRtStr_Type;
  s->length = length;
  s->data = data;
  return (PyRtObject *)s;
}

int64_t pyrt_str_length(PyRtObject *o)
{
  return ((PyRtStrObject *)o)->length;
}

PyRtObject *pyrt_str_item(PyRtObject *o, int64_t i)
{
  PyRtStrObject *s = (PyRtStrObject *)o;
  if (i < 0 || i >= s->length)
    PYRT_RAISE("IndexError: string index out of range");
  char *one = __ESBMC_alloca(2);
  one[0] = s->data[i];
  one[1] = 0;
  return pyrt_str_new(one, 1);
}

PyRtObject *pyrt_str_concat(PyRtObject *a, PyRtObject *b)
{
  if (!pyrt_str_check(b))
    PYRT_RAISE("TypeError: can only concatenate str to str");
  PyRtStrObject *x = (PyRtStrObject *)a;
  PyRtStrObject *y = (PyRtStrObject *)b;
  int64_t total = x->length + y->length;
  if (total >= PYRT_STR_CAPACITY)
    PYRT_RAISE("pyrt: string longer than the model holds");

  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < x->length; ++i)
    buffer[i] = x->data[i];
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < y->length; ++i)
    buffer[x->length + i] = y->data[i];
  buffer[total] = 0;
  return pyrt_str_new(buffer, total);
}

/* Equality only: CPython also orders strings, which this model does not, so
 * an ordered comparison falls through to a TypeError. */
PyRtObject *pyrt_str_richcompare(PyRtObject *a, PyRtObject *b, int op)
{
  if (!pyrt_str_check(b) || (op != Py_EQ && op != Py_NE))
    return &pyrt_NotImplemented;
  PyRtStrObject *x = (PyRtStrObject *)a;
  PyRtStrObject *y = (PyRtStrObject *)b;
  bool equal = x->length == y->length;
  #pragma unroll
  for (int64_t i = 0; equal && i < PYRT_STR_CAPACITY && i < x->length; ++i)
    equal = x->data[i] == y->data[i];
  return pyrt_bool_from(op == Py_EQ ? equal : !equal);
}

static bool pyrt_str_space(char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

static PyRtObject *pyrt_str_sub(const char *data, int64_t from, int64_t n)
{
  __ESBMC_assert(
    n < PYRT_STR_CAPACITY, "pyrt: string longer than the model holds");
  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < n; ++i)
    buffer[i] = data[from + i];
  buffer[n] = 0;
  return pyrt_str_new(buffer, n);
}

/* Whether needle occurs in haystack starting at `at`. */
static bool pyrt_str_match(
  const char *hay,
  int64_t hn,
  const char *needle,
  int64_t nn,
  int64_t at)
{
  if (at + nn > hn)
    return false;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < nn; ++i)
    if (hay[at + i] != needle[i])
      return false;
  return true;
}

/* chars == 0 means whitespace, matching str.strip()'s default. */
static bool pyrt_str_stripped(char c, PyRtStrObject *chars)
{
  if (!chars)
    return pyrt_str_space(c);
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < chars->length; ++i)
    if (chars->data[i] == c)
      return true;
  return false;
}

/* ASCII only, which is what this model's bytes-backed str can express. */
PyRtObject *pyrt_strmeth_case(PyRtObject *o, int mode)
{
  PyRtStrObject *s = (PyRtStrObject *)o;
  __ESBMC_assert(
    s->length < PYRT_STR_CAPACITY, "pyrt: string longer than the model holds");
  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY + 1);
  bool at_word_start = true;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY && i < s->length; ++i)
  {
    char c = s->data[i];
    bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (mode == 1 || (mode == 2 && at_word_start))
      c = (c >= 'a' && c <= 'z') ? (char)(c - 32) : c;
    else
      c = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    buffer[i] = c;
    at_word_start = !alpha;
  }
  buffer[s->length] = 0;
  return pyrt_str_new(buffer, s->length);
}

PyRtObject *pyrt_strmeth_affix(PyRtObject *o, PyRtObject *affix, bool at_end)
{
  if (!pyrt_str_check(affix))
    PYRT_RAISE("TypeError: this argument must be str");
  PyRtStrObject *s = (PyRtStrObject *)o;
  PyRtStrObject *a = (PyRtStrObject *)affix;
  if (a->length > s->length)
    return pyrt_bool_from(false);
  int64_t at = at_end ? s->length - a->length : 0;
  return pyrt_bool_from(
    pyrt_str_match(s->data, s->length, a->data, a->length, at));
}

/* Leftmost match, or the rightmost one when from_right; -1 when absent. */
int64_t pyrt_str_search(PyRtStrObject *s, PyRtStrObject *sub, bool from_right)
{
  int64_t found = -1;
  #pragma unroll
  for (int64_t i = 0; i < PYRT_STR_CAPACITY; ++i)
  {
    if (i + sub->length > s->length)
      break;
    if (pyrt_str_match(s->data, s->length, sub->data, sub->length, i))
    {
      found = i;
      if (!from_right)
        break;
    }
  }
  return found;
}

PyRtObject *pyrt_strmeth_find(PyRtObject *o, PyRtObject *sub, bool from_right)
{
  if (!pyrt_str_check(sub))
    PYRT_RAISE("TypeError: find() argument must be str");
  return pyrt_long_from(
    pyrt_str_search((PyRtStrObject *)o, (PyRtStrObject *)sub, from_right));
}

PyRtObject *pyrt_strmeth_strindex(PyRtObject *o, PyRtObject *sub)
{
  if (!pyrt_str_check(sub))
    PYRT_RAISE("TypeError: index() argument must be str");
  int64_t at =
    pyrt_str_search((PyRtStrObject *)o, (PyRtStrObject *)sub, false);
  if (at < 0)
    PYRT_RAISE("ValueError: substring not found");
  return pyrt_long_from(at);
}

/* Non-overlapping, as CPython counts: "aaa".count("aa") is 1. */
PyRtObject *pyrt_strmeth_strcount(PyRtObject *o, PyRtObject *sub)
{
  if (!pyrt_str_check(sub))
    PYRT_RAISE("TypeError: count() argument must be str");
  PyRtStrObject *s = (PyRtStrObject *)o;
  PyRtStrObject *t = (PyRtStrObject *)sub;
  if (t->length == 0)
    return pyrt_long_from(s->length + 1);
  int64_t seen = 0;
  int64_t at = 0;
  #pragma unroll
  for (int64_t guard = 0; guard < PYRT_STR_CAPACITY; ++guard)
  {
    if (at + t->length > s->length)
      break;
    if (pyrt_str_match(s->data, s->length, t->data, t->length, at))
    {
      ++seen;
      at += t->length;
    }
    else
      ++at;
  }
  return pyrt_long_from(seen);
}

/* sep == 0 is str.split()'s no-argument form: split on runs of whitespace and
 * drop empty fields, which is a different rule from splitting on a separator
 * (CPython: "  a  ".split() == ["a"], but "a,,".split(",") == ["a","",""]). */
PyRtObject *pyrt_strmeth_split(PyRtObject *o, PyRtObject *sep)
{
  PyRtStrObject *s = (PyRtStrObject *)o;
  PyRtObject *result = pyrt_list_new();

  /* str.split(None) is the no-argument form, not a separator of that name. */
  if (!sep || sep == &pyrt_None)
  {
    int64_t start = -1;
    #pragma unroll
    for (int64_t i = 0; i < PYRT_STR_CAPACITY && i <= s->length; ++i)
    {
      bool boundary = i == s->length || pyrt_str_space(s->data[i]);
      if (!boundary && start < 0)
        start = i;
      else if (boundary && start >= 0)
      {
        pyrt_list_append(result, pyrt_str_sub(s->data, start, i - start));
        start = -1;
      }
    }
    return result;
  }

  if (!pyrt_str_check(sep))
    PYRT_RAISE("TypeError: split() separator must be str");
  PyRtStrObject *d = (PyRtStrObject *)sep;
  if (d->length == 0)
    PYRT_RAISE("ValueError: empty separator");

  int64_t start = 0;
  int64_t at = 0;
  #pragma unroll
  for (int64_t guard = 0; guard < PYRT_STR_CAPACITY; ++guard)
  {
    if (at + d->length > s->length)
      break;
    if (pyrt_str_match(s->data, s->length, d->data, d->length, at))
    {
      pyrt_list_append(result, pyrt_str_sub(s->data, start, at - start));
      at += d->length;
      start = at;
    }
    else
      ++at;
  }
  pyrt_list_append(result, pyrt_str_sub(s->data, start, s->length - start));
  return result;
}

PyRtObject *pyrt_strmeth_join(PyRtObject *sep, PyRtObject *iterable)
{
  PyRtStrObject *d = (PyRtStrObject *)sep;
  int64_t count = pyrt_iter_length(iterable);
  /* One past the capacity so the terminator still fits when the result fills
     the buffer exactly. */
  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY + 1);
  int64_t length = 0;

  #pragma unroll
  for (int64_t i = 0; i < PYRT_LIST_CAPACITY && i < count; ++i)
  {
    if (i > 0)
    {
      #pragma unroll
      for (int64_t k = 0; k < PYRT_STR_CAPACITY && k < d->length; ++k)
      {
        __ESBMC_assert(
          length < PYRT_STR_CAPACITY,
          "pyrt: string longer than the model holds");
        buffer[length++] = d->data[k];
      }
    }

    PyRtObject *item = pyrt_iter_item(iterable, i);
    if (!pyrt_str_check(item))
      PYRT_RAISE("TypeError: sequence item is not str");
    PyRtStrObject *piece = (PyRtStrObject *)item;
    #pragma unroll
    for (int64_t k = 0; k < PYRT_STR_CAPACITY && k < piece->length; ++k)
    {
      __ESBMC_assert(
        length < PYRT_STR_CAPACITY, "pyrt: string longer than the model holds");
      buffer[length++] = piece->data[k];
    }
  }
  buffer[length] = 0;
  return pyrt_str_new(buffer, length);
}

PyRtObject *
pyrt_strmeth_replace(PyRtObject *o, PyRtObject *old, PyRtObject *rep)
{
  if (!pyrt_str_check(old) || !pyrt_str_check(rep))
    PYRT_RAISE("TypeError: replace() arguments must be str");
  PyRtStrObject *s = (PyRtStrObject *)o;
  PyRtStrObject *a = (PyRtStrObject *)old;
  PyRtStrObject *b = (PyRtStrObject *)rep;

  char *buffer = __ESBMC_alloca(PYRT_STR_CAPACITY + 1);
  int64_t length = 0;
  int64_t at = 0;

  #pragma unroll
  for (int64_t guard = 0; guard < PYRT_STR_CAPACITY && at <= s->length; ++guard)
  {
    /* An empty `old` matches before every character and once at the end,
     * so "ab".replace("", "-") is "-a-b-" in CPython. */
    bool hit = a->length == 0
                 ? true
                 : pyrt_str_match(s->data, s->length, a->data, a->length, at);
    if (hit)
    {
      #pragma unroll
      for (int64_t k = 0; k < PYRT_STR_CAPACITY && k < b->length; ++k)
      {
        __ESBMC_assert(
          length < PYRT_STR_CAPACITY,
          "pyrt: string longer than the model holds");
        buffer[length++] = b->data[k];
      }
      at += a->length;
      if (a->length > 0)
        continue;
    }
    if (at >= s->length)
      break;
    __ESBMC_assert(
      length < PYRT_STR_CAPACITY, "pyrt: string longer than the model holds");
    buffer[length++] = s->data[at++];
  }
  buffer[length] = 0;
  return pyrt_str_new(buffer, length);
}

PyRtObject *
pyrt_strmeth_strip(PyRtObject *o, PyRtObject *chars, bool left, bool right)
{
  if (chars && !pyrt_str_check(chars))
    PYRT_RAISE("TypeError: strip() argument must be str");
  PyRtStrObject *s = (PyRtStrObject *)o;
  PyRtStrObject *set = (PyRtStrObject *)chars;

  int64_t lo = 0;
  int64_t hi = s->length;
  if (left)
  {
    #pragma unroll
    for (int64_t i = 0; i < PYRT_STR_CAPACITY && lo < hi; ++i)
    {
      if (!pyrt_str_stripped(s->data[lo], set))
        break;
      ++lo;
    }
  }
  if (right)
  {
    #pragma unroll
    for (int64_t i = 0; i < PYRT_STR_CAPACITY && hi > lo; ++i)
    {
      if (!pyrt_str_stripped(s->data[hi - 1], set))
        break;
      --hi;
    }
  }
  return pyrt_str_sub(s->data, lo, hi - lo);
}
