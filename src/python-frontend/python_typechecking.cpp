#include <python-frontend/python_typechecking.h>

#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <unordered_set>
#include <util/std_types.h>

python_typechecking::python_typechecking(python_converter &converter)
  : converter_(converter)
{
}

std::vector<typet> python_typechecking::collect_annotation_types(
  const nlohmann::json &annotation) const
{
  std::vector<typet> collected;
  if (annotation.is_null() || !annotation.contains("_type"))
    return collected;

  type_handler &type_handler = converter_.get_type_handler();
  python_dict_handler *dict_handler = converter_.get_dict_handler();

  std::function<void(const nlohmann::json &)> collect =
    [&](const nlohmann::json &node) {
      if (node.is_null() || !node.contains("_type"))
        return;

      const std::string node_type = node["_type"].get<std::string>();

      if (
        node_type == "BinOp" && node.contains("op") &&
        node["op"].contains("_type") && node["op"]["_type"] == "BitOr")
      {
        collect(node["left"]);
        collect(node["right"]);
        return;
      }

      if (
        node_type == "Subscript" && node.contains("value") &&
        node["value"].contains("id"))
      {
        const std::string base = node["value"]["id"].get<std::string>();
        if (base == "Union")
        {
          if (node.contains("slice"))
          {
            const auto &slice = node["slice"];
            if (slice.contains("elts"))
            {
              for (const auto &elt : slice["elts"])
                collect(elt);
            }
            else
              collect(slice);
          }
          return;
        }
        if (base == "Optional")
        {
          if (node.contains("slice"))
            collect(node["slice"]);
          collected.push_back(none_type());
          return;
        }
        if (base == "List" || base == "list")
        {
          collected.push_back(type_handler.get_list_type());
          return;
        }
        if (base == "Dict" || base == "dict")
        {
          collected.push_back(dict_handler->get_dict_struct_type());
          return;
        }
      }

      try
      {
        collected.push_back(converter_.get_type_from_annotation(node, node));
      }
      catch (const std::exception &)
      {
      }
    };

  collect(annotation);

  std::unordered_set<typet, irep_full_hash, irep_full_eq> seen_types;
  std::vector<typet> unique_types;
  unique_types.reserve(collected.size());
  for (const auto &type : collected)
  {
    if (seen_types.insert(type).second)
      unique_types.push_back(type);
  }
  return unique_types;
}

void python_typechecking::cache_annotation_types(
  symbolt &symbol,
  const nlohmann::json &annotation)
{
  if (annotation.is_null())
    return;
  auto key = symbol.id.as_string();
  if (annotation_type_cache_.find(key) != annotation_type_cache_.end())
  {
    if (symbol.python_annotation_types.empty())
      symbol.python_annotation_types = annotation_type_cache_.at(key);
    return;
  }

  auto types = collect_annotation_types(annotation);
  if (!types.empty())
  {
    annotation_type_cache_[key] = types;
    symbol.python_annotation_types = types;
  }
}

std::vector<typet>
python_typechecking::get_annotation_types(const std::string &symbol_id) const
{
  auto it = annotation_type_cache_.find(symbol_id);
  if (it != annotation_type_cache_.end())
    return it->second;
  return {};
}

bool python_typechecking::should_skip_type_assertion(
  const typet &annotated_type) const
{
  if (annotated_type.is_nil() || annotated_type.id().empty())
    return true;

  if (annotated_type.id() == "empty")
    return true;

  if (
    annotated_type.id() == "pointer" &&
    annotated_type.subtype().id() == "empty")
    return true; // Any

  if (annotated_type.id() == "struct")
  {
    const struct_typet &struct_type = to_struct_type(annotated_type);
    const std::string &tag = struct_type.tag().as_string();
    if (tag.rfind("tag-Optional_", 0) == 0)
      return true;
  }

  return false;
}

exprt python_typechecking::build_isinstance_check(
  const exprt &value_expr,
  const typet &annotated_type) const
{
  if (should_skip_type_assertion(annotated_type))
    return nil_exprt();

  typet effective_type = annotated_type;

  const namespacet &ns = converter_.name_space();

  if (effective_type.id() == "pointer")
  {
    typet pointed_type = effective_type.subtype();

    if (pointed_type.is_symbol())
    {
      const symbolt *symbol = ns.lookup(pointed_type);
      if (symbol != nullptr)
        pointed_type = symbol->type;
    }

    if (pointed_type.is_struct())
      effective_type = pointed_type;
  }

  if (effective_type.id() == "empty")
    return nil_exprt();

  exprt type_operand;
  if (effective_type.is_symbol())
  {
    const symbolt *symbol = ns.lookup(effective_type);
    if (symbol == nullptr)
      return nil_exprt();
    type_operand = symbol_expr(*symbol);
  }
  else
    type_operand = gen_zero(effective_type);

  exprt isinstance_expr("isinstance", typet("bool"));
  isinstance_expr.copy_to_operands(value_expr);
  isinstance_expr.move_to_operands(type_operand);
  return isinstance_expr;
}

bool python_typechecking::build_type_assertion(
  const exprt &value_expr,
  const typet &annotated_type,
  const std::vector<typet> &allowed_types,
  const std::string &context_name,
  const locationt &location,
  code_assertt &out_assert) const
{
  std::vector<typet> effective_types = allowed_types;
  if (effective_types.empty() && !should_skip_type_assertion(annotated_type))
    effective_types.push_back(annotated_type);

  std::vector<exprt> checks;
  for (const auto &type : effective_types)
  {
    if (type == none_type())
      continue;

    exprt isinstance_expr = build_isinstance_check(value_expr, type);
    if (!isinstance_expr.is_nil())
      checks.push_back(isinstance_expr);
  }

  if (checks.empty())
    return false;

  exprt assertion_expr = checks.front();
  for (size_t i = 1; i < checks.size(); ++i)
  {
    exprt or_expr("or", typet("bool"));
    or_expr.move_to_operands(assertion_expr);
    or_expr.move_to_operands(checks[i]);
    assertion_expr = or_expr;
  }

  out_assert = code_assertt(assertion_expr);
  out_assert.location() = location;

  if (!context_name.empty())
  {
    type_handler &type_handler = converter_.get_type_handler();

    std::string type_label = type_handler.type_to_string(annotated_type);
    if (type_label.empty())
    {
      if (annotated_type.id() == "struct")
      {
        const auto &struct_type = to_struct_type(annotated_type);
        std::string tag = struct_type.tag().as_string();
        type_label = tag.rfind("tag-", 0) == 0 ? tag.substr(4) : tag;
      }
      else if (annotated_type.id() == "symbol")
      {
        type_label = annotated_type.identifier().as_string();
      }
    }

    if (!type_label.empty())
    {
      out_assert.location().comment(
        "Expected '" + context_name + "' to be of type '" + type_label + "'");
    }
    else
    {
      out_assert.location().comment(
        "Type annotation check for '" + context_name + "'");
    }
  }

  return true;
}

void python_typechecking::emit_type_annotation_assertion(
  const exprt &value_expr,
  const typet &annotated_type,
  const std::vector<typet> &allowed_types,
  const std::string &context_name,
  const locationt &location,
  codet &target_block)
{
  code_assertt type_assert;
  if (!build_type_assertion(
        value_expr,
        annotated_type,
        allowed_types,
        context_name,
        location,
        type_assert))
    return;

  target_block.copy_to_operands(type_assert);
}

void python_typechecking::inject_parameter_type_assertions(
  const nlohmann::json &function_node,
  const symbol_id &function_id,
  const code_typet &type,
  exprt &function_body)
{
  if (function_body.statement() != "block")
    return;

  if (!function_node.contains("args") || !function_node["args"].is_object())
    return;

  const auto &args_node = function_node["args"];
  if (
    !args_node.contains("args") && !args_node.contains("posonlyargs") &&
    !args_node.contains("kwonlyargs"))
    return;

  std::unordered_map<std::string, typet> param_types;
  for (const auto &arg : type.arguments())
  {
    const std::string base = arg.get_base_name().as_string();
    if (!base.empty())
      param_types.emplace(base, arg.type());
  }

  std::vector<code_assertt> assertions;
  auto try_append_assert = [&](const nlohmann::json &arg_json) {
    if (
      !arg_json.contains("annotation") || arg_json["annotation"].is_null() ||
      !arg_json.contains("arg"))
      return;

    std::string arg_name = arg_json["arg"].get<std::string>();
    if (arg_name.empty() || arg_name == "self" || arg_name == "cls")
      return;

    auto type_it = param_types.find(arg_name);
    if (type_it == param_types.end())
      return;

    std::string sid = function_id.to_string() + "@" + arg_name;
    symbol_exprt param_expr(sid, type_it->second);
    locationt loc = converter_.get_location_from_decl(arg_json);
    std::vector<typet> allowed_types = get_annotation_types(sid);

    code_assertt type_assert;
    if (build_type_assertion(
          param_expr,
          type_it->second,
          allowed_types,
          arg_name,
          loc,
          type_assert))
      assertions.push_back(type_assert);
  };

  auto process_arg_array = [&](const nlohmann::json &arr) {
    if (!arr.is_array())
      return;
    for (const auto &elem : arr)
      try_append_assert(elem);
  };

  if (args_node.contains("posonlyargs"))
    process_arg_array(args_node["posonlyargs"]);
  if (args_node.contains("args"))
    process_arg_array(args_node["args"]);
  if (args_node.contains("kwonlyargs"))
    process_arg_array(args_node["kwonlyargs"]);

  if (assertions.empty())
    return;

  code_blockt &block = static_cast<code_blockt &>(function_body);
  auto &ops = block.operands();
  ops.insert(ops.begin(), assertions.begin(), assertions.end());
}

std::string
python_typechecking::get_constructor_name(const nlohmann::json &func_node) const
{
  if (!func_node.contains("_type"))
    return {};

  if (func_node["_type"] == "Name" && func_node.contains("id"))
    return func_node["id"].get<std::string>();

  if (func_node["_type"] == "Attribute" && func_node.contains("attr"))
    return func_node["attr"].get<std::string>();

  return {};
}

bool python_typechecking::class_derives_from(
  const std::string &class_name,
  const std::string &expected_base) const
{
  return converter_.get_type_handler().class_derives_from(
    class_name, expected_base);
}
