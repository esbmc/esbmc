#include "pyrt.h"

/* The two roots of the exception hierarchy, so `class MyError(Exception)`
 * resolves and isinstance walks from a user exception up to Exception and
 * BaseException. They carry no slots of their own: an exception is an ordinary
 * object here, and what makes it catchable is the raise, not its type. */
PyRtTypeObject PyRtBaseException_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "BaseException",
  .tp_base = &PyRtObject_Type};

PyRtTypeObject PyRtException_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "Exception",
  .tp_base = &PyRtBaseException_Type};

/* CPython's shape, so `except LookupError:` catches a KeyError and
 * `except ArithmeticError:` a ZeroDivisionError. */
PyRtTypeObject PyRtLookupError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "LookupError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtKeyError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "KeyError",
  .tp_base = &PyRtLookupError_Type};

PyRtTypeObject PyRtIndexError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "IndexError",
  .tp_base = &PyRtLookupError_Type};

PyRtTypeObject PyRtArithmeticError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "ArithmeticError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtZeroDivisionError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "ZeroDivisionError",
  .tp_base = &PyRtArithmeticError_Type};

PyRtTypeObject PyRtTypeError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "TypeError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtValueError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "ValueError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtAttributeError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "AttributeError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtNameError_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "NameError",
  .tp_base = &PyRtException_Type};

PyRtTypeObject PyRtStopIteration_Type = {
  .ob_type = &PyRtType_Type,
  .tp_name = "StopIteration",
  .tp_base = &PyRtException_Type};

/* An error the models raise has nowhere to throw from: C has no throw, so the
 * function records it here and returns, and the caller turns it into one.
 * That is how CPython gets an error out of a C function. Null when there is
 * none in flight. */
PyRtObject *pyrt_pending = 0;

void pyrt_set_pending(PyRtTypeObject *cls)
{
  PyRtInstanceObject *o = __ESBMC_alloca(sizeof(PyRtInstanceObject));
  o->ob_type = cls;
  o->attrs.size = 0;
  pyrt_pending = (PyRtObject *)o;
}

PyRtObject *pyrt_take_pending(void)
{
  PyRtObject *raised = pyrt_pending;
  pyrt_pending = 0;
  return raised;
}
