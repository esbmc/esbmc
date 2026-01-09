#include <python-frontend/exception_utils.h>

#include <util/c_types.h>

namespace python_exception_utils
{
exprt make_exception_raise(
  const type_handler &type_handler,
  const std::string &exc,
  const std::string &message,
  const locationt *location)
{
  if (!type_utils::is_python_exceptions(exc))
  {
    log_error("This exception type is not supported: {}", exc);
    abort();
  }

  typet type = type_handler.get_typet(exc);

  exprt size = constant_exprt(
    integer2binary(message.size(), bv_width(size_type())),
    integer2string(message.size()),
    size_type());
  typet t = array_typet(char_type(), size);
  string_constantt string_name(message, t, string_constantt::k_default);

  exprt sym("struct", type);
  sym.copy_to_operands(address_of_exprt(string_name));

  exprt raise = side_effect_exprt("cpp-throw", type);
  raise.move_to_operands(sym);
  if (location != nullptr)
  {
    raise.location() = *location;
    raise.location().user_provided(true);
  }

  return raise;
}
} // namespace python_exception_utils
