#include "python_list_internal.h"
#include <python-frontend/string/string_handler.h>

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

// The type hash an untyped push of an int records.
exprt python_list::typed_int_type_id() const
{
  constant_exprt id(size_type());
  id.set_value(integer2binary(
    std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(typed_int_type())),
    config.ansi_c.address_width));
  return id;
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

exprt python_list::typed_extend(const exprt &list, const exprt &source)
{
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tlist_extend"));
  call.arguments() = {list, source};
  call.type() = empty_typet();
  call.location() = converter_.get_location_from_decl(list_value_);
  return call;
}

exprt python_list::typed_compare(
  const exprt &lhs,
  const exprt &rhs,
  bool negate)
{
  const locationt location = converter_.get_location_from_decl(list_value_);
  symbolt &eq = converter_.create_tmp_symbol(
    list_value_, "$tlist_eq$", bool_type(), exprt());
  code_declt decl(build_symbol(eq));
  decl.location() = location;
  converter_.add_instruction(decl);

  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tlist_eq"));
  call.arguments() = {lhs, rhs};
  call.lhs() = build_symbol(eq);
  call.type() = bool_type();
  call.location() = location;
  converter_.add_instruction(call);
  return negate ? exprt(not_exprt(build_symbol(eq))) : build_symbol(eq);
}

exprt python_list::typed_or_untyped_compare(
  const exprt &lhs,
  const exprt &rhs,
  const std::string &op)
{
  // The gate flags a comparison only for == and != of two typed lists.
  if (is_typed(list_value_))
    return typed_compare(lhs, rhs, op == "NotEq");
  return compare(lhs, rhs, op);
}

exprt python_list::typed_binop()
{
  // `a + b`, `[x] * n` or `n * [x]`, operands evaluated left to right. A
  // constructor call in an operand must not take the outer target as its self.
  const bool concat = list_value_["op"]["_type"] == "Add";
  exprt *saved_lhs = converter_.current_lhs;
  converter_.current_lhs = nullptr;
  std::vector<exprt> args;
  for (const char *side : {"left", "right"})
  {
    const nlohmann::json &operand = list_value_[side];
    const bool repeated = !concat && operand["_type"] == "List";
    args.push_back(
      repeated
        ? typed_int_temp(operand, converter_.get_expr(operand["elts"][0]))
        : converter_.get_expr(operand));
  }
  converter_.current_lhs = saved_lhs;

  if (concat)
    return typed_new_list("c:@F@__ESBMC_tlist_concat", args);
  if (list_value_["right"]["_type"] == "List")
    std::swap(args[0], args[1]);
  args[1] =
    build_typecast(to_value_expr(args[1], converter_.ns), typed_int_type());
  return typed_new_list("c:@F@__ESBMC_tlist_repeat", args);
}

bool python_list::is_typed_call(const nlohmann::json &call)
{
  const std::string func = call["func"].value("id", "");
  if (func == "__ESBMC_str_codes")
    return true;
  if (call.value("esbmc_typed_dict", false))
    return func == "__ESBMC_tdict_set" || func == "__ESBMC_tdict_in" ||
           func == "__ESBMC_tdict_max";
  return (is_typed(call) &&
          (func == "_nondet_list_int" || func == "max" || func == "min")) ||
         ((is_typed(call) || call.value("esbmc_typed_list_source", false)) &&
          func == "sorted");
}

exprt python_list::typed_call(const std::string &func_name)
{
  if (func_name == "sorted")
    return typed_sorted();
  if (func_name == "max" || func_name == "min")
    return typed_extremum(func_name);
  if (func_name == "__ESBMC_tdict_set")
    return typed_dict_set();
  if (func_name == "__ESBMC_tdict_in")
    return typed_dict_in();
  if (func_name == "__ESBMC_tdict_max")
    return typed_dict_max();
  if (func_name == "__ESBMC_str_codes")
    return typed_str_codes();
  return typed_nondet(converter_.get_expr(list_value_["args"][0]));
}

exprt python_list::typed_sorted()
{
  bool reverse = false;
  for (const auto &keyword : list_value_["keywords"])
    reverse = keyword["value"]["value"].get<bool>();
  const exprt sorted = typed_new_list(
    "c:@F@__ESBMC_tlist_sorted",
    {converter_.get_expr(list_value_["args"][0]),
     reverse ? exprt(true_exprt()) : exprt(false_exprt())});
  if (is_typed(list_value_))
    return sorted;

  // The gate found a use of the result the typed model does not implement:
  // hand it an untyped list holding the same ints.
  return typed_new_list(
    "c:@F@__ESBMC_tlist_to_list", {sorted, typed_int_type_id()});
}

exprt python_list::typed_extremum(const std::string &func_name)
{
  // max(x) or min(x) of a typed x: CPython's ValueError on an empty x, else
  // the extremum of its slots.
  const exprt list = converter_.get_expr(list_value_["args"][0]);
  const locationt location = converter_.get_location_from_decl(list_value_);
  codet throw_code("expression");
  throw_code.operands().push_back(
    converter_.get_exception_handler().gen_exception_raise(
      "ValueError", func_name + "() arg is an empty sequence"));
  code_ifthenelset guard;
  guard.cond() = build_equal(
    build_deref_member(list, "size", size_type()), gen_zero(size_type()));
  guard.then_case() = throw_code;
  guard.location() = location;
  converter_.add_instruction(guard);

  const typet int_type = typed_int_type();
  symbolt &result = converter_.create_tmp_symbol(
    list_value_, "$tlist_extremum$", int_type, exprt());
  code_declt decl(build_symbol(result));
  decl.location() = location;
  converter_.add_instruction(decl);
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tlist_extremum"));
  call.arguments() = {
    list, func_name == "min" ? exprt(true_exprt()) : exprt(false_exprt())};
  call.lhs() = build_symbol(result);
  call.type() = int_type;
  call.location() = location;
  converter_.add_instruction(call);
  return build_symbol(result);
}

bool python_list::is_typed_dict_read(const nlohmann::json &node)
{
  return node.is_object() && node.value("esbmc_typed_dict", false);
}

std::vector<exprt>
python_list::typed_dict_key(const nlohmann::json &dict, const json_range &key)
{
  // The dict, the key's arity, and its one or two ints, each evaluated once
  // in order.
  std::vector<exprt> args{converter_.get_expr(dict), exprt()};
  for (const nlohmann::json &part : key)
    args.push_back(typed_int_temp(part, converter_.get_expr(part)));
  args[1] = from_integer(args.size() - 2, size_type());
  if (args.size() == 3)
    args.push_back(from_integer(0, typed_int_type()));
  return args;
}

exprt python_list::typed_dict_slot(const std::vector<exprt> &args)
{
  const locationt location = converter_.get_location_from_decl(list_value_);
  const typet slot_type =
    pointer_typet(symbol_typet("tag-struct __ESBMC_PySlot"));
  symbolt &slot = converter_.create_tmp_symbol(
    list_value_, "$tdict_slot$", slot_type, exprt());
  code_declt decl(build_symbol(slot));
  decl.location() = location;
  converter_.add_instruction(decl);
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tdict_at"));
  call.arguments() = args;
  call.lhs() = build_symbol(slot);
  call.type() = slot_type;
  call.location() = location;
  converter_.add_instruction(call);
  return build_symbol(slot);
}

exprt python_list::typed_dict_get()
{
  // d[k] of a typed dict, whose key the gate wrote as a tuple of its ints:
  // KeyError when k is absent, else its value.
  const nlohmann::json &key = list_value_["slice"]["elts"];
  const std::vector<exprt> args =
    typed_dict_key(list_value_["value"], json_range(key.begin(), key.end()));
  // A Counter reads 0 for a missing key and does not insert it.
  if (list_value_.value("esbmc_typed_dict_default", false))
    return typed_dict_value("c:@F@__ESBMC_tdict_get0", args);
  const exprt slot = typed_dict_slot(args);
  codet throw_code("expression");
  throw_code.operands().push_back(
    converter_.get_exception_handler().gen_exception_raise(
      "KeyError", "key not found in dictionary"));
  code_ifthenelset guard;
  guard.cond() = build_equal(slot, gen_zero(slot.type()));
  guard.then_case() = throw_code;
  guard.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(guard);
  return build_deref_member(slot, "v", typed_int_type());
}

exprt python_list::typed_dict_value(
  const char *model,
  const std::vector<exprt> &args)
{
  const locationt location = converter_.get_location_from_decl(list_value_);
  const typet int_type = typed_int_type();
  symbolt &value = converter_.create_tmp_symbol(
    list_value_, "$tdict_value$", int_type, exprt());
  code_declt decl(build_symbol(value));
  decl.location() = location;
  converter_.add_instruction(decl);
  code_function_callt call;
  call.function() = build_symbol(model_function(converter_, model));
  call.arguments() = args;
  call.lhs() = build_symbol(value);
  call.type() = int_type;
  call.location() = location;
  converter_.add_instruction(call);
  return build_symbol(value);
}

exprt python_list::typed_dict_max()
{
  // __ESBMC_tdict_max(d, arity) for `max(d.values())`: CPython's ValueError
  // on an empty dict, else its largest value.
  const nlohmann::json &args = list_value_["args"];
  const exprt dict = converter_.get_expr(args[0]);
  codet throw_code("expression");
  throw_code.operands().push_back(
    converter_.get_exception_handler().gen_exception_raise(
      "ValueError", "max() arg is an empty sequence"));
  code_ifthenelset guard;
  guard.cond() = build_equal(
    build_deref_member(dict, "size", size_type()), gen_zero(size_type()));
  guard.then_case() = throw_code;
  guard.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(guard);
  return typed_dict_value(
    "c:@F@__ESBMC_tdict_max",
    {dict, from_integer(args[1]["value"].get<long long>(), size_type())});
}

exprt python_list::typed_dict_in()
{
  // __ESBMC_tdict_in(d, *key) for `k in d`.
  const nlohmann::json &args = list_value_["args"];
  const exprt slot = typed_dict_slot(
    typed_dict_key(args[0], json_range(args.begin() + 1, args.end())));
  return not_exprt(build_equal(slot, gen_zero(slot.type())));
}

exprt python_list::typed_dict_set()
{
  // __ESBMC_tdict_set(d, v, *key) for `d[k] = v`: the value is evaluated
  // before the key, as CPython does.
  const nlohmann::json &args = list_value_["args"];
  const exprt value = typed_int_temp(args[1], converter_.get_expr(args[1]));
  std::vector<exprt> call_args =
    typed_dict_key(args[0], json_range(args.begin() + 2, args.end()));
  call_args.insert(call_args.begin() + 1, value);
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tdict_set"));
  call.arguments() = call_args;
  call.type() = empty_typet();
  call.location() = converter_.get_location_from_decl(list_value_);
  return call;
}

exprt python_list::typed_str_codes()
{
  // __ESBMC_str_codes(nondet_str()): the codes of the nondet buffer's
  // characters, its length the buffer's own length symbol. The copy covers
  // the buffer's whole capacity, so no character is dropped.
  const exprt str = converter_.get_expr(list_value_["args"][0]);
  const int capacity = string_handler::nondet_str_buffer_size() - 1;
  const auto &lengths = converter_.input_str_to_len_sym_;
  const bool buffer = str.is_address_of() && str.op0().id() == "index" &&
                      str.op0().op0().is_symbol();
  auto it = buffer ? lengths.find(str.op0().op0().identifier().as_string())
                   : lengths.end();
  if (it == lengths.end() || capacity > 64)
    throw std::runtime_error("unsupported: a str held as character codes");
  const symbolt *length = converter_.find_symbol(it->second);
  const exprt list = typed_new_list(
    "c:@F@__ESBMC_tlist_from_chars",
    {str,
     build_typecast(build_symbol(*length), size_type()),
     from_integer(capacity, size_type())});
  if (is_typed(list_value_))
    return list;
  // The gate left the codes untyped: an untyped list of the same ints.
  return typed_new_list(
    "c:@F@__ESBMC_tlist_to_list", {list, typed_int_type_id()});
}

bool python_list::reads_typed(const nlohmann::json &node)
{
  return node.is_object() && node.value("esbmc_typed_source", false);
}

bool python_list::is_typed_subscript(const nlohmann::json &node)
{
  return is_typed(node) || reads_typed(node);
}

exprt python_list::typed_subscript(
  const exprt &list,
  const nlohmann::json &slice)
{
  if (reads_typed(list_value_))
    return typed_slice(list, slice);
  const exprt element = typed_element(list, converter_.get_expr(slice));
  // A bool list holds 0 or 1 in each slot; a store writes the slot itself.
  if (
    list_value_.value("esbmc_typed_bool", false) &&
    list_value_["ctx"].value("_type", "") == "Load")
    return build_typecast(element, bool_type());
  return element;
}

exprt python_list::typed_copy_call(
  const exprt &list,
  const std::vector<exprt> &bounds,
  long long step)
{
  std::vector<exprt> args{list};
  args.insert(args.end(), bounds.begin(), bounds.end());
  args.push_back(from_integer(step, typed_int_type()));
  if (is_typed(list_value_))
    return typed_new_list("c:@F@__ESBMC_tlist_slice", args);

  // The gate left the result untyped: an untyped list whose every element
  // is an int, which its readers must know.
  args.push_back(typed_int_type_id());
  const exprt result = typed_new_list("c:@F@__ESBMC_tlist_slice_untyped", args);
  elem_types().assign(result.identifier().as_string(), typed_int_type());
  return result;
}

long long python_list::typed_slice_step(const nlohmann::json &slice)
{
  // The gate admits only a missing step or an int literal, maybe negated.
  long long step = 1;
  if (slice.contains("step") && !slice["step"].is_null())
  {
    const nlohmann::json &literal = slice["step"]["_type"] == "UnaryOp"
                                      ? slice["step"]["operand"]
                                      : slice["step"];
    step = literal["value"].get<long long>();
    if (slice["step"]["_type"] == "UnaryOp")
      step = -step;
  }
  // The raise diverts control, so what is built after it is dead.
  emit_slice_zero_step_raise(slice, step == 0);
  return step == 0 ? 1 : step;
}

std::vector<exprt> python_list::typed_slice_bounds(const nlohmann::json &slice)
{
  // Each bound is passed as (given, value). The model computes a missing
  // bound's default itself.
  std::vector<exprt> bounds;
  for (const char *name : {"lower", "upper"})
  {
    const bool given = slice.contains(name) && !slice[name].is_null();
    bounds.push_back(given ? exprt(true_exprt()) : exprt(false_exprt()));
    bounds.push_back(
      given ? typed_int_temp(slice[name], converter_.get_expr(slice[name]))
            : from_integer(0, typed_int_type()));
  }
  return bounds;
}

exprt python_list::typed_slice(const exprt &list, const nlohmann::json &slice)
{
  const long long step = typed_slice_step(slice);
  return typed_copy_call(list, typed_slice_bounds(slice), step);
}

void python_list::assign_slice(
  const exprt &list,
  const nlohmann::json &slice,
  const nlohmann::json &value)
{
  if (!is_typed(list_value_))
    return handle_slice_assignment(list, slice, value);
  typed_slice_assign(
    list.type().is_pointer() ? list : build_address_of(list), slice, value);
}

void python_list::typed_slice_assign(
  const exprt &list,
  const nlohmann::json &slice,
  const nlohmann::json &value)
{
  // The value is a typed list, a fresh one, or reversed() of a list or a
  // slice, which is its copy with step -1. It is evaluated before the
  // target's slice, as in CPython.
  const bool reversed =
    value["_type"] == "Call" && value["func"].value("id", "") == "reversed";
  exprt source = converter_.get_expr(reversed ? value["args"][0] : value);
  const exprt missing = from_integer(0, typed_int_type());
  if (reversed)
    source = typed_new_list(
      "c:@F@__ESBMC_tlist_slice",
      {source,
       false_exprt(),
       missing,
       false_exprt(),
       missing,
       from_integer(-1, typed_int_type())});

  const long long step = typed_slice_step(slice);
  code_function_callt call;
  call.function() =
    build_symbol(model_function(converter_, "c:@F@__ESBMC_tlist_slice_assign"));
  call.arguments() = {list};
  for (const exprt &bound : typed_slice_bounds(slice))
    call.arguments().push_back(bound);
  call.arguments().push_back(from_integer(step, typed_int_type()));
  call.arguments().push_back(source);
  call.type() = empty_typet();
  call.location() = converter_.get_location_from_decl(slice);
  converter_.add_instruction(call);
}

exprt python_list::typed_copy(const exprt &list)
{
  const exprt missing = from_integer(0, typed_int_type());
  return typed_copy_call(
    list, {false_exprt(), missing, false_exprt(), missing}, 1);
}
