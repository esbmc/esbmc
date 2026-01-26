#ifndef PYTHON_FRONTEND_EXCEPTION_UTILS_H
#define PYTHON_FRONTEND_EXCEPTION_UTILS_H

#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <util/location.h>
#include <util/message.h>
#include <util/std_expr.h>
#include <util/string_constant.h>

#include <string>

namespace python_exception_utils
{
exprt make_exception_raise(
  const type_handler &type_handler,
  const std::string &exc,
  const std::string &message,
  const locationt *location = nullptr);
} // namespace python_exception_utils

#endif // PYTHON_FRONTEND_EXCEPTION_UTILS_H
