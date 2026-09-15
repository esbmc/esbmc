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

typedef struct __pyrt_list
{
  PyRt_HEAD;
  int64_t size;
  PyRtObject **items;
} PyRtListObject;

typedef PyRtObject *(*unaryfunc)(PyRtObject *);
typedef PyRtObject *(*binaryfunc)(PyRtObject *, PyRtObject *);
typedef int64_t (*lenfunc)(PyRtObject *);
typedef PyRtObject *(*ssizeargfunc)(PyRtObject *, int64_t);
typedef void (*ssizeobjargproc)(PyRtObject *, int64_t, PyRtObject *);
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

struct __pyrt_type
{
  PyRt_HEAD;
  const char *tp_name;
  PyRtTypeObject *tp_base;
  PyRtNumberMethods *tp_as_number;
  PyRtSequenceMethods *tp_as_sequence;
  richcmpfunc tp_richcompare;
};

#define PYRT_RAISE(msg)                                                        \
  do                                                                           \
  {                                                                            \
    __ESBMC_assert(0, msg);                                                    \
    __ESBMC_assume(0);                                                         \
  } while (0)

#define PYRT_LIST_CAPACITY 64

extern PyRtTypeObject PyRtType_Type;
extern PyRtTypeObject PyRtNone_Type;
extern PyRtTypeObject PyRtNotImplemented_Type;
extern PyRtTypeObject PyRtLong_Type;
extern PyRtTypeObject PyRtBool_Type;
extern PyRtTypeObject PyRtList_Type;

extern PyRtObject pyrt_None;
extern PyRtObject pyrt_NotImplemented;
extern PyRtLongObject pyrt_True;
extern PyRtLongObject pyrt_False;

PyRtObject *pyrt_bool_from(bool b);
bool pyrt_long_check(PyRtObject *o);
PyRtObject *pyrt_long_from(int64_t v);
PyRtObject *pyrt_list_new(void);

bool pyrt_is_true(PyRtObject *o);
PyRtObject *pyrt_number_add(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_subtract(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_multiply(PyRtObject *a, PyRtObject *b);
PyRtObject *pyrt_number_negative(PyRtObject *o);
PyRtObject *pyrt_richcompare(PyRtObject *a, PyRtObject *b, int op);
PyRtObject *pyrt_builtin_len(PyRtObject *o);
PyRtObject *pyrt_getitem(PyRtObject *o, PyRtObject *key);
void pyrt_setitem(PyRtObject *o, PyRtObject *key, PyRtObject *value);
void pyrt_list_append(PyRtObject *o, PyRtObject *value);

#endif
