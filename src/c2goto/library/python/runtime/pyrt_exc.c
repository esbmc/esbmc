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
