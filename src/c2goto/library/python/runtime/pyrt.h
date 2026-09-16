#ifndef PYRT_H
#define PYRT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct __pyrt_type PyRtTypeObject;

/* The type pointer is the literal first member of every object: symex treats a
 * struct as a prefix of another only when member names and types match, which
 * a nested `PyRtObject ob_base` member would not. */
#define PyRt_HEAD PyRtTypeObject *ob_type

typedef struct __pyrt_object
{
  PyRt_HEAD;
} PyRtObject;

/* int and bool share this layout; bools exist only as pyrt_True/False. */
typedef struct __pyrt_long
{
  PyRt_HEAD;
  int64_t value;
} PyRtLongObject;

/* Immutable; `data` points at a literal emitted by the frontend or at a
 * buffer this model allocated. */
typedef struct __pyrt_str
{
  PyRt_HEAD;
  int64_t length;
  const char *data;
} PyRtStrObject;

typedef struct __pyrt_float
{
  PyRt_HEAD;
  double value;
} PyRtFloatObject;

typedef struct __pyrt_list
{
  PyRt_HEAD;
  int64_t size;
  PyRtObject **items;
} PyRtListObject;

#define PYRT_ATTRS_CAPACITY 16

/* Attribute names are interned (pyrt_names.c and the frontend), so they
 * compare by address. */
typedef struct __pyrt_attrs
{
  int64_t size;
  const char *names[PYRT_ATTRS_CAPACITY];
  PyRtObject *values[PYRT_ATTRS_CAPACITY];
} PyRtAttrs;

typedef struct __pyrt_instance
{
  PyRt_HEAD;
  PyRtAttrs *attrs;
} PyRtInstanceObject;

#define PYRT_MAX_ARGS 6

/* Positional arguments, passed by value in one fixed-width struct so a single
 * pointer type calls any function. Separate fields keep each argument's value
 * set apart, which an array read through a pointer would merge. */
typedef struct __pyrt_args
{
  PyRtObject *a0, *a1, *a2, *a3, *a4, *a5;
} PyRtArgs;

typedef PyRtObject *(*pyrt_code)(PyRtArgs args);

typedef struct __pyrt_function
{
  PyRt_HEAD;
  int64_t arity;
  pyrt_code code;
} PyRtFunctionObject;

typedef struct __pyrt_method
{
  PyRt_HEAD;
  PyRtObject *self;
  PyRtObject *function;
} PyRtMethodObject;

/* Insertion-ordered, with keys compared by == rather than hashed. */
typedef struct __pyrt_dict
{
  PyRt_HEAD;
  int64_t size;
  int64_t *hashes;
  PyRtObject **keys;
  PyRtObject **values;
} PyRtDictObject;

typedef PyRtObject *(*unaryfunc)(PyRtObject *);
typedef PyRtObject *(*binaryfunc)(PyRtObject *, PyRtObject *);
typedef int64_t (*lenfunc)(PyRtObject *);
typedef PyRtObject *(*ssizeargfunc)(PyRtObject *, int64_t);
typedef void (*ssizeobjargproc)(PyRtObject *, int64_t, PyRtObject *);
typedef void (*objobjargproc)(PyRtObject *, PyRtObject *, PyRtObject *);
typedef PyRtObject *(*richcmpfunc)(PyRtObject *, PyRtObject *, int);
typedef bool (*inquiry)(PyRtObject *);

#define Py_LT 0
#define Py_LE 1
#define Py_EQ 2
#define Py_NE 3
#define Py_GT 4
#define Py_GE 5

typedef struct __pyrt_number_methods
{
  binaryfunc nb_add;
  binaryfunc nb_subtract;
  binaryfunc nb_multiply;
  binaryfunc nb_true_divide;
  binaryfunc nb_floor_divide;
  binaryfunc nb_remainder;
  binaryfunc nb_power;
  unaryfunc nb_negative;
  inquiry nb_bool;
} PyRtNumberMethods;

typedef struct __pyrt_sequence_methods
{
  lenfunc sq_length;
  binaryfunc sq_concat;
  ssizeargfunc sq_item;
  ssizeobjargproc sq_ass_item;
} PyRtSequenceMethods;

typedef struct __pyrt_mapping_methods
{
  binaryfunc mp_subscript;
  objobjargproc mp_ass_subscript;
} PyRtMappingMethods;

#define PYRT_TPFLAGS_HEAPTYPE 1u
#define PYRT_TPFLAGS_READY 2u
#define PYRT_TPFLAGS_OWNS_TABLES 4u

struct __pyrt_type
{
  PyRt_HEAD;
  const char *tp_name;
  PyRtTypeObject *tp_base;
  PyRtNumberMethods *tp_as_number;
  PyRtSequenceMethods *tp_as_sequence;
  PyRtMappingMethods *tp_as_mapping;
  richcmpfunc tp_richcompare;
  PyRtAttrs *tp_attrs;
  /* Direct subclasses, threaded through their tp_sibling. */
  PyRtTypeObject *tp_subclass;
  PyRtTypeObject *tp_sibling;
  unsigned tp_flags;
};

#define PYRT_RAISE(msg)                                                        \
  do                                                                           \
  {                                                                            \
    __ESBMC_assert(0, msg);                                                    \
    __ESBMC_assume(0);                                                         \
  } while (0)

#define PYRT_LIST_CAPACITY 64
#define PYRT_MAX_CLASSES 16
#define PYRT_STR_CAPACITY 64
#define PYRT_DICT_CAPACITY 16
#define PYRT_POW_BOUND 64

extern PyRtTypeObject PyRtType_Type;
extern PyRtTypeObject PyRtObject_Type;
extern PyRtTypeObject PyRtNone_Type;
extern PyRtTypeObject PyRtNotImplemented_Type;
extern PyRtTypeObject PyRtLong_Type;
extern PyRtTypeObject PyRtBool_Type;
extern PyRtTypeObject PyRtList_Type;
extern PyRtTypeObject PyRtStr_Type;
extern PyRtTypeObject PyRtFloat_Type;
extern PyRtTypeObject PyRtDict_Type;
extern PyRtTypeObject PyRtFunction_Type;
extern PyRtTypeObject PyRtMethod_Type;

extern PyRtObject pyrt_None;
extern PyRtObject pyrt_NotImplemented;
extern PyRtLongObject pyrt_True;
extern PyRtLongObject pyrt_False;

extern const char pyrt_str___init__[];
extern const char pyrt_str___add__[];
extern const char pyrt_str___radd__[];
extern const char pyrt_str___sub__[];
extern const char pyrt_str___rsub__[];
extern const char pyrt_str___mul__[];
extern const char pyrt_str___rmul__[];
extern const char pyrt_str___neg__[];
extern const char pyrt_str___bool__[];
extern const char pyrt_str___len__[];
extern const char pyrt_str___getitem__[];
extern const char pyrt_str___setitem__[];
extern const char pyrt_str___lt__[];
extern const char pyrt_str___le__[];
extern const char pyrt_str___eq__[];
extern const char pyrt_str___ne__[];
extern const char pyrt_str___gt__[];
extern const char pyrt_str___ge__[];
extern const char pyrt_str_append[];
extern const char pyrt_str___class__[];

PyRtObject *pyrt_bool_from(bool b);
bool pyrt_long_check(PyRtObject *o);
PyRtObject *pyrt_long_from(int64_t v);
PyRtObject *pyrt_long_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_true_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_floor_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_remainder(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_power(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_long_richcompare(PyRtObject *a, PyRtObject *b, int op);
PyRtObject *pyrt_list_new(void);
PyRtObject *pyrt_str_new(const char *data, int64_t length);
bool pyrt_float_check(PyRtObject *o);
PyRtObject *pyrt_float_from(double v);
double pyrt_number_as_double(PyRtObject *o);
PyRtObject *pyrt_dict_new(void);
bool pyrt_key_equal(PyRtObject *a, PyRtObject *b);
int64_t pyrt_key_hash(PyRtObject *key);
int64_t pyrt_dict_find(PyRtDictObject *d, PyRtObject *key);
PyRtObject *pyrt_dict_key_at(PyRtObject *o, int64_t i);
int64_t pyrt_iter_length(PyRtObject *o);
PyRtObject *pyrt_iter_item(PyRtObject *o, int64_t i);
PyRtObject *pyrt_contains(PyRtObject *container, PyRtObject *item);
int64_t pyrt_as_index(PyRtObject *o);
PyRtObject *pyrt_nondet_bool(void);
PyRtObject *pyrt_nondet_int(void);

bool pyrt_is_true(PyRtObject *o);
PyRtObject *pyrt_number_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_true_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_floor_divide(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_remainder(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_power(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_negative(PyRtObject *o);
PyRtObject *pyrt_richcompare(PyRtObject *a, PyRtObject *b, int op);
PyRtObject *pyrt_builtin_len(PyRtObject *o);
PyRtObject *pyrt_builtin_abs(PyRtObject *o);
PyRtObject *pyrt_builtin_all(PyRtObject *o);
PyRtObject *pyrt_builtin_any(PyRtObject *o);
PyRtObject *pyrt_builtin_sum(PyRtObject *o);
PyRtObject *pyrt_builtin_sum_start(PyRtObject *o, PyRtObject *start);
PyRtObject *pyrt_builtin_min2(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_builtin_max2(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_builtin_min_iter(PyRtObject *o);
PyRtObject *pyrt_builtin_max_iter(PyRtObject *o);
PyRtObject *pyrt_getitem(PyRtObject *o, PyRtObject *key);
void pyrt_setitem(PyRtObject *o, PyRtObject *key, PyRtObject *value);
void pyrt_list_append(PyRtObject *o, PyRtObject *value);

PyRtObject *pyrt_attrs_find(PyRtAttrs *attrs, const char *name);
void pyrt_attrs_set(PyRtAttrs *attrs, const char *name, PyRtObject *value);
PyRtObject *pyrt_type_lookup(PyRtTypeObject *t, const char *name);
void pyrt_type_ready(PyRtTypeObject *t);
void pyrt_update_slot(PyRtTypeObject *t, const char *name);

PyRtObject *pyrt_call(PyRtObject *callable, PyRtArgs args, int64_t nargs);
PyRtObject *pyrt_call_function(
  PyRtObject *callable,
  PyRtArgs args,
  int64_t nargs);
PyRtObject *pyrt_call_method(
  PyRtObject *o,
  const char *name,
  PyRtArgs args,
  int64_t nargs);
PyRtObject *pyrt_getattr(PyRtObject *o, const char *name);
PyRtObject *pyrt_type_of(PyRtObject *o);
bool pyrt_isinstance(PyRtObject *o, PyRtObject *cls);
bool pyrt_hasattr(PyRtObject *o, const char *name);
void pyrt_setattr(PyRtObject *o, const char *name, PyRtObject *value);

#endif
