#include "python_list_internal.h"

using namespace python_expr;

exprt python_list::build_set_membership_call(
  const symbolt &set,
  const nlohmann::json &op,
  const exprt &elem,
  const std::string &method_name)
{
  const std::string c_func = "c:@F@__ESBMC_set_" + method_name;
  const symbolt *func = converter_.symbol_table().find_symbol(c_func);
  if (!func)
    throw std::runtime_error(c_func + " function not found in symbol table");

  list_elem_info elem_info = get_list_element_info(op, elem);

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  code_function_callt call;
  call.function() = build_symbol(*func);
  call.arguments().push_back(build_symbol(set));
  call.arguments().push_back(element_arg);
  call.arguments().push_back(build_symbol(*elem_info.elem_type_sym));
  call.arguments().push_back(elem_info.elem_size);
  call.type() = bool_type();
  call.location() = elem_info.location;

  // Track the new element's compile-time type info so subsequent ops
  // (`elem in set`, set comparisons) recognise it.
  if (method_name == "add")
    add_type_info(
      set.id.as_string(),
      elem_info.elem_symbol->id.as_string(),
      elem_info.elem_symbol->get_type());

  return converter_.convert_expression_to_code(call);
}
