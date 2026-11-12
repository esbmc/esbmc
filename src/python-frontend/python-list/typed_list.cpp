#include "python_list_internal.h"

using namespace python_expr;

// Typed lists (#10451). The gate in preprocessor/typed_lists.py flags the
// nodes of a list whose every element is an int and whose items no other code
// reads; their elements live in the list's int64 `slot` array instead of one
// untyped cell each (see the __ESBMC_tlist_* functions in python/list.c).
// Each flagged operation is lowered by one function below, called from that
// operation's usual dispatch site. A new operation adds a gate rule, a model
// function and a lowering here.

bool python_list::is_typed(const nlohmann::json &node)
{
  return node.is_object() && node.value("esbmc_typed_list", false);
}

static const symbolt &
model_function(const python_converter &converter, const char *name)
{
  const symbolt *symbol = converter.symbol_table().find_symbol(name);
  assert(symbol && "typed list model function not found");
  return *symbol;
}

typet python_list::typed_int_type() const
{
  return converter_.get_type_handler().get_typet("int", 0);
}

exprt python_list::typed_int_temp(
  const nlohmann::json &node,
  const exprt &value)
{
  const typet int_type = typed_int_type();
  const locationt location = converter_.get_location_from_decl(node);
  symbolt &tmp =
    converter_.create_tmp_symbol(node, "$tlist_elem$", int_type, exprt());
  code_declt decl(build_symbol(tmp));
  decl.location() = location;
  converter_.add_instruction(decl);
  code_assignt assign(
    build_symbol(tmp),
    build_typecast(to_value_expr(value, converter_.ns), int_type));
  assign.location() = location;
  converter_.add_instruction(assign);
  return build_symbol(tmp);
}

exprt python_list::typed_new_list(
  const char *create,
  const std::vector<exprt> &args)
{
  const locationt location = converter_.get_location_from_decl(list_value_);
  const typet list_type = converter_.get_type_handler().get_list_type();
  symbolt &list =
    converter_.create_tmp_symbol(list_value_, "$py_list$", list_type, exprt());
  code_declt decl(build_symbol(list));
  decl.location() = location;
  converter_.add_instruction(decl);

  code_function_callt call;
  call.function() = build_symbol(model_function(converter_, create));
  call.arguments() = args;
  call.lhs() = build_symbol(list);
  call.type() = list_type;
  call.location() = location;
  converter_.add_instruction(call);
  return build_symbol(list);
}

code_function_callt python_list::typed_push(
  const nlohmann::json &node,
  const exprt &list,
  const exprt &value)
{
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tlist_push"));
  call.arguments().push_back(list);
  call.arguments().push_back(typed_int_temp(node, value));
  call.type() = empty_typet();
  call.location() = converter_.get_location_from_decl(node);
  return call;
}

exprt python_list::typed_literal()
{
  const exprt list = typed_new_list("c:@F@__ESBMC_tlist_create", {});
  for (const auto &elt : list_value_["elts"])
  {
    // A constructor call inside an element must not take the outer target as
    // its self.
    exprt *saved_lhs = converter_.current_lhs;
    converter_.current_lhs = nullptr;
    const exprt value = converter_.get_expr(elt);
    converter_.current_lhs = saved_lhs;
    converter_.add_instruction(typed_push(elt, list, value));
  }
  return list;
}

exprt python_list::typed_nondet(const exprt &max_size)
{
  // The builder's elements are nondet_int()s, which never return the smallest
  // int unless ints are unbounded (#9923).
  const bool exclude_min = !config.options.get_bool_option("int-encoding");
  return typed_new_list(
    "c:@F@__ESBMC_tlist_nondet",
    {build_typecast(to_value_expr(max_size, converter_.ns), typed_int_type()),
     exclude_min ? exprt(true_exprt()) : exprt(false_exprt())});
}

exprt python_list::typed_element(const exprt &list, const exprt &index)
{
  // The same index normalisation and IndexError as an untyped list, then
  // the element itself as an lvalue, so a store needs nothing else.
  const typet slot_type = symbol_typet("tag-struct __ESBMC_PySlot");
  const element_storage storage{
    "c:@F@__ESBMC_tlist_at", "slot", pointer_typet(slot_type)};
  return build_deref_member(
    build_list_at_call(list, index, list_value_, storage),
    "v",
    typed_int_type());
}

exprt python_list::typed_append(const exprt &list, const exprt &value)
{
  return typed_push(list_value_, list, value);
}
