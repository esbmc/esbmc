#include <python-frontend/function_call/expr.h>

#include <python-frontend/json_utils.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_set.h>
#include <python-frontend/tuple_handler.h>
#include <unordered_map>
#include <unordered_set>

namespace
{
bool has_supported_method(
  const std::unordered_set<std::string> &supported_methods,
  const std::string &method_name)
{
  return supported_methods.find(method_name) != supported_methods.end();
}

bool is_supported_set_method(const std::string &method_name)
{
  static const std::unordered_set<std::string> supported_methods = {
    "add", "discard", "issubset", "update", "symmetric_difference"};
  return has_supported_method(supported_methods, method_name);
}

bool is_supported_list_method(const std::string &method_name)
{
  static const std::unordered_set<std::string> supported_methods = {
    "append",
    "pop",
    "insert",
    "remove",
    "clear",
    "extend",
    "copy",
    "sort",
    "reverse"};
  return has_supported_method(supported_methods, method_name);
}

bool is_supported_dict_method(const std::string &method_name)
{
  static const std::unordered_set<std::string> supported_methods = {
    "get", "setdefault", "update", "pop", "popitem", "copy"};
  return has_supported_method(supported_methods, method_name);
}

bool is_set_membership_mutator(const std::string &method_name)
{
  return method_name == "add" || method_name == "discard";
}

bool is_supported_dict_class_method(const std::string &method_name)
{
  return method_name == "fromkeys";
}
} // namespace

const symbolt *
function_call_expr::get_object_list_symbol(std::string &display_name) const
{
  const auto &func_value = call_["func"]["value"];

  // Subscript case: e.g. nested[0].append(99) — resolve the inner list symbol
  // via the compile-time list_type_map rather than through a plain name lookup.
  if (func_value["_type"] == "Subscript")
  {
    const auto &base_node = func_value["value"];
    if (!base_node.contains("id"))
      return nullptr;

    std::string base_name = base_node["id"].get<std::string>();
    base_name = json_utils::get_object_alias(converter_.ast(), base_name);

    symbol_id base_sym_id = converter_.create_symbol_id();
    base_sym_id.set_object(base_name);
    const symbolt *base_sym = converter_.find_symbol(base_sym_id.to_string());
    if (!base_sym)
      return nullptr;

    const auto &slice_node = func_value["slice"];
    const typet list_type = converter_.get_type_handler().get_list_type();
    const std::string &base_id = base_sym->id.as_string();

    // Nested list (list-of-list) path: only meaningful when the base is itself
    // a list, since list_type_map keys list-of-list types. For non-list bases
    // (e.g. dicts whose value is a list) we fall through to the get_expr
    // dispatch below, which can also resolve dict-subscript receivers.
    if (base_sym->type == list_type)
    {
      // Constant index: resolve directly from list_type_map.
      if (
        slice_node["_type"] == "Constant" &&
        slice_node["value"].is_number_integer())
      {
        const size_t index = slice_node["value"].get<size_t>();

        if (python_list::get_list_element_type(base_id, index) != list_type)
          return nullptr;

        const std::string inner_id =
          python_list::get_list_element_id(base_id, index);
        if (inner_id.empty())
          return nullptr;

        display_name = base_name + "[" + std::to_string(index) + "]";
        return converter_.find_symbol(inner_id);
      }

      // Non-constant index (e.g. nested[i].append(v)): delegate to the existing
      // subscript handler. For comprehension-generated nested lists the handler
      // hits the list_type_map early-return path and yields the template inner
      // list symbol (the element produced inside the loop body) without emitting
      // any runtime instructions.
      const exprt subscript_expr = converter_.get_expr(func_value);
      if (subscript_expr.is_symbol())
      {
        const symbolt *sym =
          converter_.find_symbol(subscript_expr.identifier().as_string());
        if (sym && sym->type == list_type)
        {
          const std::string idx_str = slice_node.contains("id")
                                        ? slice_node["id"].get<std::string>()
                                        : "(expr)";
          display_name = base_name + "[" + idx_str + "]";
          return sym;
        }
      }
      return nullptr;
    }

    // Dict subscript whose value is a list (e.g. d[k].append(v) where
    // d is a dict[K, list[V]] or a defaultdict(list)). The dict-subscript
    // expression returns the stored PyListObject pointer; wrap it in a
    // temp symbol so list method handlers can treat it as a named list.
    // List mutations through the temp alias the dict slot because lists in
    // the Python model are reference-typed (PyListObject *).
    const exprt subscript_expr = converter_.get_expr(func_value);
    if (subscript_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$dict_list$", list_type, subscript_expr);
      std::string idx_str = "(expr)";
      if (slice_node.contains("id"))
        idx_str = slice_node["id"].get<std::string>();
      else if (
        slice_node["_type"] == "Constant" &&
        slice_node["value"].is_number_integer())
        idx_str = std::to_string(slice_node["value"].get<size_t>());
      display_name = base_name + "[" + idx_str + "]";
      return &tmp;
    }

    return nullptr;
  }

  // Attribute case: e.g. obj.mutable_attr.append(1)
  if (func_value["_type"] == "Attribute")
  {
    const exprt attr_expr = converter_.get_expr(func_value);
    const typet list_type = converter_.get_type_handler().get_list_type();

    if (
      func_value.contains("value") && func_value["value"].contains("id") &&
      func_value.contains("attr"))
    {
      display_name = func_value["value"]["id"].get<std::string>() + "." +
                     func_value["attr"].get<std::string>();
    }

    if (attr_expr.is_symbol())
    {
      const symbolt *sym =
        converter_.find_symbol(attr_expr.identifier().as_string());
      if (sym && sym->type == list_type)
        return sym;
    }

    if (attr_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$attr_list$", list_type, attr_expr);
      return &tmp;
    }

    return nullptr;
  }

  // Call case: e.g. a.setdefault(k, []).append(99)
  if (func_value["_type"] == "Call")
  {
    const exprt call_expr = converter_.get_expr(func_value);
    const typet list_type = converter_.get_type_handler().get_list_type();
    if (call_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$call_list$", list_type, call_expr);
      display_name = "$call_list$";
      return &tmp;
    }
    return nullptr;
  }

  // Plain name case: e.g. mylist.append(99)
  display_name = get_object_name();
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(display_name);
  return converter_.find_symbol(list_symbol_id.to_string());
}

void function_call_expr::materialize_list_symbol(const symbolt *sym) const
{
  if (!sym || sym->value.is_nil())
    return;

  const std::string &name = sym->name.as_string();
  if (
    name.find("$attr_list$") == std::string::npos &&
    name.find("$call_list$") == std::string::npos &&
    name.find("$dict_list$") == std::string::npos)
    return;

  code_declt decl(symbol_expr(*sym));
  decl.copy_to_operands(sym->value);
  decl.location() = sym->location;
  converter_.current_block->copy_to_operands(decl);
}

const symbolt *function_call_expr::resolve_list_symbol_or_throw(
  std::string &display_name) const
{
  const symbolt *list_symbol = get_object_list_symbol(display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + display_name);

  return list_symbol;
}

const symbolt *function_call_expr::find_required_symbol(
  const std::string &id,
  const std::string &message) const
{
  const symbolt *symbol = converter_.symbol_table().find_symbol(id);
  if (!symbol)
    throw std::runtime_error(message);
  return symbol;
}

bool function_call_expr::has_binop_receiver() const
{
  return call_.contains("func") && call_["func"].is_object() &&
         call_["func"].contains("value") &&
         call_["func"]["value"].is_object() &&
         call_["func"]["value"].contains("_type") &&
         call_["func"]["value"]["_type"] == "BinOp";
}

bool function_call_expr::has_attribute_receiver() const
{
  return call_.contains("func") && call_["func"].is_object() &&
         call_["func"].contains("_type") &&
         call_["func"]["_type"] == "Attribute";
}

bool function_call_expr::is_ambiguous_list_dict_method(
  const std::string &method_name) const
{
  return method_name == "pop" || method_name == "copy";
}

bool function_call_expr::receiver_is_list_symbol() const
{
  std::string dummy;
  const symbolt *sym = get_object_list_symbol(dummy);
  const typet list_type = type_handler_.get_list_type();
  return sym != nullptr && sym->type == list_type;
}

bool function_call_expr::receiver_is_set_symbol() const
{
  std::string dummy;
  const symbolt *sym = get_object_list_symbol(dummy);
  return sym != nullptr && sym->is_set;
}

const symbolt *
function_call_expr::resolve_set_symbol_or_throw(std::string &display_name) const
{
  const symbolt *set_symbol = get_object_list_symbol(display_name);
  materialize_list_symbol(set_symbol);

  if (!set_symbol)
    throw std::runtime_error("Set variable not found: " + display_name);

  return set_symbol;
}

exprt function_call_expr::resolve_dict_expr_or_throw() const
{
  std::string dict_name = get_object_name();
  if (!dict_name.empty())
  {
    symbol_id dict_symbol_id = converter_.create_symbol_id();
    dict_symbol_id.set_object(dict_name);
    const symbolt *dict_symbol =
      converter_.find_symbol(dict_symbol_id.to_string());
    if (!dict_symbol)
      throw std::runtime_error("Dictionary variable not found: " + dict_name);
    return symbol_expr(*dict_symbol);
  }

  exprt literal = converter_.get_expr(call_["func"]["value"]);
  symbolt &tmp =
    converter_.create_tmp_symbol(call_, "$dict_lit$", literal.type(), exprt());
  converter_.add_instruction(code_declt(symbol_expr(tmp)));
  converter_.add_instruction(code_assignt(symbol_expr(tmp), literal));
  return symbol_expr(tmp);
}

exprt function_call_expr::handle_list_insert() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("insert() takes exactly two arguments");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  exprt index_expr = converter_.get_expr(args[0]);
  exprt value_to_insert = converter_.get_expr(args[1]);

  if (value_to_insert.is_constant())
  {
    symbolt &insert_value_symbol = converter_.create_tmp_symbol(
      call_, "insert_value", size_type(), gen_zero(size_type()));
    code_declt insert_value(symbol_expr(insert_value_symbol));
    insert_value.copy_to_operands(value_to_insert);
    converter_.current_block->copy_to_operands(insert_value);
  }

  python_list list(converter_, nlohmann::json());
  list.add_type_info(
    list_symbol->id.as_string(),
    value_to_insert.identifier().as_string(),
    value_to_insert.type());

  return list.build_insert_list_call(
    *list_symbol, index_expr, call_, value_to_insert);
}

exprt function_call_expr::handle_list_clear() const
{
  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  const symbolt *clear_func = find_required_symbol(
    "c:@F@__ESBMC_list_clear",
    "__ESBMC_list_clear function not found in symbol table");

  code_function_callt clear_call;
  clear_call.function() = symbol_expr(*clear_func);
  clear_call.arguments().push_back(symbol_expr(*list_symbol));
  clear_call.type() = empty_typet();
  clear_call.location() = converter_.get_location_from_decl(call_);

  return clear_call;
}

exprt function_call_expr::handle_list_pop() const
{
  const auto &args = call_["args"];

  if (args.size() > 1)
    throw std::runtime_error("pop() takes at most 1 argument");

  const symbolt *list_symbol = nullptr;

  if (has_binop_receiver())
  {
    exprt list_expr = converter_.get_expr(call_["func"]["value"]);
    if (list_expr.is_symbol())
    {
      list_symbol = converter_.symbol_table().find_symbol(
        to_symbol_expr(list_expr).get_identifier());
    }
  }

  if (!list_symbol)
  {
    std::string list_display_name;
    list_symbol = resolve_list_symbol_or_throw(list_display_name);
  }

  exprt index_expr;
  if (args.empty())
    index_expr = from_integer(-1, signedbv_typet(64));
  else
    index_expr = converter_.get_expr(args[0]);

  python_list list_helper(converter_, call_);
  return list_helper.build_pop_list_call(*list_symbol, index_expr, call_);
}

bool function_call_expr::is_tuple_method_call() const
{
  if (!has_attribute_receiver())
    return false;

  const std::string &method_name = function_id_.get_function();
  if (method_name != "count" && method_name != "index")
    return false;

  exprt receiver = converter_.get_expr(call_["func"]["value"]);
  return converter_.get_tuple_handler().is_tuple_type(receiver.type());
}

exprt function_call_expr::handle_tuple_method() const
{
  const std::string &method_name = function_id_.get_function();
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error(
      "tuple." + method_name + "() takes exactly one argument");

  exprt receiver = converter_.get_expr(call_["func"]["value"]);
  const struct_typet &tuple_type = to_struct_type(receiver.type());
  const auto &components = tuple_type.components();

  exprt elem = converter_.get_expr(args[0]);

  if (method_name == "count")
  {
    typet result_type = int_type();
    exprt total = gen_zero(result_type);
    for (const auto &comp : components)
    {
      exprt member = member_exprt(receiver, comp.get_name(), comp.type());
      exprt eq = equality_exprt(member, elem);
      if_exprt sel(eq, gen_one(result_type), gen_zero(result_type));
      sel.type() = result_type;
      total = plus_exprt(total, sel);
      total.type() = result_type;
    }
    return total;
  }

  if (components.empty())
    throw std::runtime_error("tuple.index() on empty tuple");

  typet result_type = int_type();
  exprt any_match = gen_boolean(false);
  for (const auto &comp : components)
  {
    exprt member = member_exprt(receiver, comp.get_name(), comp.type());
    exprt eq = equality_exprt(member, elem);
    any_match = or_exprt(any_match, eq);
  }
  code_assertt found_assert(any_match);
  found_assert.location() = converter_.get_location_from_decl(call_);
  found_assert.location().comment("ValueError: tuple.index(x): x not in tuple");
  converter_.add_instruction(found_assert);

  size_t n = components.size();
  exprt result = from_integer(BigInt(n - 1), result_type);
  for (size_t k = n - 1; k-- > 0;)
  {
    exprt member =
      member_exprt(receiver, components[k].get_name(), components[k].type());
    exprt eq = equality_exprt(member, elem);
    if_exprt sel(eq, from_integer(BigInt(k), result_type), result);
    sel.type() = result_type;
    result = sel;
  }
  return result;
}

bool function_call_expr::is_dict_method_call() const
{
  if (!has_attribute_receiver())
    return false;

  const std::string &method_name = function_id_.get_function();

  if (!is_supported_dict_method(method_name))
    return false;

  if (method_name == "pop")
  {
    return !receiver_is_list_symbol();
  }

  return true;
}

exprt function_call_expr::handle_dict_method() const
{
  const std::string &method_name = function_id_.get_function();
  using dict_method_handlert =
    exprt (python_dict_handler::*)(const exprt &, const nlohmann::json &);
  exprt dict_expr = resolve_dict_expr_or_throw();

  static const std::unordered_map<std::string, dict_method_handlert>
    method_dispatch = {
      {"get", &python_dict_handler::handle_dict_get},
      {"setdefault", &python_dict_handler::handle_dict_setdefault},
      {"update", &python_dict_handler::handle_dict_update},
      {"pop", &python_dict_handler::handle_dict_pop},
      {"popitem", &python_dict_handler::handle_dict_popitem},
      {"copy", &python_dict_handler::handle_dict_copy},
    };

  const auto it = method_dispatch.find(method_name);
  if (it != method_dispatch.end())
    return (converter_.get_dict_handler()->*(it->second))(dict_expr, call_);

  throw std::runtime_error("Unsupported dict method: " + method_name);
}

bool function_call_expr::is_dict_class_method_call() const
{
  if (!has_attribute_receiver())
    return false;
  const auto &value = call_["func"]["value"];
  if (!value.contains("_type") || value["_type"] != "Name")
    return false;
  if (value["id"] != "dict")
    return false;

  const std::string &method_name = function_id_.get_function();
  return is_supported_dict_class_method(method_name);
}

exprt function_call_expr::handle_dict_class_method() const
{
  const std::string &method_name = function_id_.get_function();

  if (is_supported_dict_class_method(method_name))
    return converter_.get_dict_handler()->handle_dict_fromkeys(call_);

  throw std::runtime_error("Unsupported dict class method: " + method_name);
}

exprt function_call_expr::handle_list_copy() const
{
  const auto &args = call_["args"];

  if (!args.empty())
    throw std::runtime_error("copy() takes no arguments");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  python_list list_helper(converter_, call_);
  return list_helper.build_copy_list_call(*list_symbol, call_);
}

exprt function_call_expr::handle_list_remove() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("remove() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  exprt value_to_remove = converter_.get_expr(args[0]);

  python_list list_helper(converter_, call_);
  return list_helper.build_remove_list_call(
    *list_symbol, call_, value_to_remove);
}

exprt function_call_expr::handle_list_sort() const
{
  const auto &args = call_["args"];
  if (!args.empty())
    throw std::runtime_error(
      "sort() positional arguments are not supported; "
      "use sort() with no arguments");

  bool reverse = false;
  if (call_.contains("keywords"))
  {
    for (const auto &kw : call_["keywords"])
    {
      const std::string name = kw.value("arg", "");
      if (name == "reverse")
      {
        exprt v = converter_.get_expr(kw["value"]);
        if (!v.is_constant())
          throw std::runtime_error(
            "sort(reverse=...) requires a constant boolean");
        reverse = v.is_true();
      }
      else
        throw std::runtime_error(
          "sort() keyword argument '" + name +
          "' is not supported (only reverse=)");
    }
  }

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  const std::string &list_id = list_symbol->id.as_string();

  int type_flag = 0;
  size_t float_type_id = 0;
  python_list::get_list_type_flags(
    list_id, converter_.get_type_handler(), type_flag, float_type_id);

  const symbolt *sort_func = find_required_symbol(
    "c:@F@__ESBMC_list_sort",
    "__ESBMC_list_sort function not found in symbol table");

  code_function_callt sort_call;
  sort_call.function() = symbol_expr(*sort_func);
  sort_call.arguments().push_back(symbol_expr(*list_symbol));
  sort_call.arguments().push_back(from_integer(type_flag, int_type()));
  sort_call.arguments().push_back(
    from_integer(float_type_id, unsignedbv_typet(config.ansi_c.address_width)));
  sort_call.type() = empty_typet();
  sort_call.location() = converter_.get_location_from_decl(call_);

  if (!reverse)
    return sort_call;

  const symbolt *reverse_func = find_required_symbol(
    "c:@F@__ESBMC_list_reverse",
    "__ESBMC_list_reverse function not found in symbol table");

  code_function_callt reverse_call;
  reverse_call.function() = symbol_expr(*reverse_func);
  reverse_call.arguments().push_back(symbol_expr(*list_symbol));
  reverse_call.type() = empty_typet();
  reverse_call.location() = converter_.get_location_from_decl(call_);

  python_list::reverse_type_info(list_id);

  code_blockt block;
  block.copy_to_operands(sort_call);
  block.copy_to_operands(reverse_call);
  return block;
}

exprt function_call_expr::handle_list_reverse() const
{
  const auto &args = call_["args"];

  if (!args.empty())
    throw std::runtime_error("reverse() takes no arguments");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  const symbolt *reverse_func = find_required_symbol(
    "c:@F@__ESBMC_list_reverse",
    "__ESBMC_list_reverse function not found in symbol table");

  code_function_callt reverse_call;
  reverse_call.function() = symbol_expr(*reverse_func);
  reverse_call.arguments().push_back(symbol_expr(*list_symbol));
  reverse_call.type() = empty_typet();
  reverse_call.location() = converter_.get_location_from_decl(call_);

  python_list::reverse_type_info(list_symbol->id.as_string());

  return reverse_call;
}

bool function_call_expr::is_set_method_call() const
{
  if (!has_attribute_receiver())
    return false;

  const std::string &method_name = function_id_.get_function();
  if (!is_supported_set_method(method_name))
    return false;

  return receiver_is_set_symbol();
}

exprt function_call_expr::handle_set_method() const
{
  const std::string &method_name = function_id_.get_function();
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error(method_name + "() takes exactly one argument");

  std::string set_display_name;
  const symbolt *set_symbol = resolve_set_symbol_or_throw(set_display_name);

  if (is_set_membership_mutator(method_name))
  {
    exprt elem = converter_.get_expr(args[0]);
    python_list helper(converter_, call_);
    return helper.build_set_membership_call(
      *set_symbol, call_, elem, method_name);
  }

  exprt other = converter_.get_expr(args[0]);
  python_set set_helper(converter_, call_);
  return set_helper.build_set_method_call(
    *set_symbol, other, call_, method_name);
}

bool function_call_expr::is_list_method_call() const
{
  if (!has_attribute_receiver())
    return false;

  const std::string &method_name = function_id_.get_function();
  if (!is_supported_list_method(method_name))
    return false;

  if (is_ambiguous_list_dict_method(method_name))
  {
    if (has_binop_receiver())
      return true;
    return receiver_is_list_symbol();
  }

  return true;
}

exprt function_call_expr::handle_list_method() const
{
  const std::string &method_name = function_id_.get_function();
  static const std::
    unordered_map<std::string, exprt (function_call_expr::*)() const>
      method_dispatch = {
        {"append", &function_call_expr::handle_list_append},
        {"insert", &function_call_expr::handle_list_insert},
        {"extend", &function_call_expr::handle_list_extend},
        {"clear", &function_call_expr::handle_list_clear},
        {"pop", &function_call_expr::handle_list_pop},
        {"copy", &function_call_expr::handle_list_copy},
        {"remove", &function_call_expr::handle_list_remove},
        {"sort", &function_call_expr::handle_list_sort},
        {"reverse", &function_call_expr::handle_list_reverse},
      };

  const auto it = method_dispatch.find(method_name);
  if (it != method_dispatch.end())
    return (this->*(it->second))();

  throw std::runtime_error("Unsupported list method: " + method_name);
}

exprt function_call_expr::handle_list_append() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("append() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  exprt value_to_append = converter_.get_expr(args[0]);

  bool is_func_call = (value_to_append.is_code() &&
                       value_to_append.get("statement") == "function_call") ||
                      (value_to_append.id() == "sideeffect" &&
                       value_to_append.get("statement") == "function_call");

  if (is_func_call)
  {
    exprt func_expr;
    exprt::operandst func_args;
    typet ret_type;

    if (value_to_append.is_code())
    {
      const code_function_callt &call =
        to_code_function_call(to_code(value_to_append));
      func_expr = call.function();
      func_args = call.arguments();
      ret_type = call.type();
    }
    else
    {
      const side_effect_expr_function_callt &call =
        to_side_effect_expr_function_call(value_to_append);
      func_expr = call.function();
      func_args = call.arguments();
      ret_type = call.type();
    }

    if (ret_type.is_nil() || ret_type.is_empty())
    {
      log_warning(
        "list.append with function call: unknown return type, assuming int");
      ret_type = int_type();
    }

    symbolt &tmp_var = converter_.create_tmp_symbol(
      call_, "$append_ret$", ret_type, gen_zero(ret_type));

    code_declt tmp_decl(symbol_expr(tmp_var));
    tmp_decl.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(tmp_decl);

    code_function_callt new_call;
    new_call.function() = func_expr;
    new_call.arguments() = func_args;
    new_call.lhs() = symbol_expr(tmp_var);
    new_call.type() = ret_type;
    new_call.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(new_call);

    value_to_append = symbol_expr(tmp_var);
  }

  if (
    value_to_append.type().is_array() &&
    value_to_append.type().subtype() == char_type())
  {
    const array_typet &array_type = to_array_type(value_to_append.type());
    if (array_type.size().is_constant())
    {
      const constant_exprt &size_const = to_constant_expr(array_type.size());
      BigInt size_value = binary2integer(size_const.value().c_str(), false);
      if (size_value == 1)
        value_to_append.type() = gen_pointer_type(char_type());
    }
  }

  if (value_to_append.is_constant())
  {
    symbolt &append_value_symbol = converter_.create_tmp_symbol(
      call_, "append_value", size_type(), gen_zero(size_type()));
    code_declt append_value(symbol_expr(append_value_symbol));
    append_value.copy_to_operands(value_to_append);
    converter_.current_block->copy_to_operands(append_value);
  }

  python_list list(converter_, nlohmann::json());

  list.add_type_info(
    list_symbol->id.as_string(),
    value_to_append.identifier().as_string(),
    value_to_append.type());

  return list.build_push_list_call(*list_symbol, call_, value_to_append);
}

exprt function_call_expr::handle_list_extend() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("extend() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = resolve_list_symbol_or_throw(list_display_name);

  exprt other_list = converter_.get_expr(args[0]);

  python_list list(converter_, nlohmann::json());

  return list.build_extend_list_call(*list_symbol, call_, other_list);
}
