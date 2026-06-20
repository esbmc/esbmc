#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/migrate.h>

#include <map>

namespace
{
// V.3: IREP2 expression-context call `fn(args...)` returning return_type
// (side_effect_function_call2tc + symbol_expr2tc), back-migrated for the legacy
// adjust/goto-convert seam; behaviour-preserving. Caller sets .location().
// Falls back to the legacy node for a dyn-sized-array return/argument type.
bool contains_dyn_array(const typet &t)
{
  if (t.is_array())
  {
    const array_typet &at = to_array_type(t);
    if (at.size().is_nil() || !at.size().is_constant())
      return true;
    return contains_dyn_array(at.subtype());
  }
  if (t.is_pointer())
    return contains_dyn_array(t.subtype());
  return false;
}

exprt build_call_expr(
  const symbolt &fn,
  const typet &return_type,
  const std::vector<exprt> &args)
{
  bool dyn = contains_dyn_array(return_type);
  for (const exprt &a : args)
    dyn = dyn || contains_dyn_array(a.type());
  if (dyn)
  {
    side_effect_expr_function_callt call;
    call.function() = symbol_expr(fn);
    for (const exprt &a : args)
      call.arguments().push_back(a);
    call.type() = return_type;
    return call;
  }
  std::vector<expr2tc> args2;
  args2.reserve(args.size());
  for (const exprt &a : args)
  {
    expr2tc a2;
    migrate_expr(a, a2);
    args2.push_back(std::move(a2));
  }
  return migrate_expr_back(side_effect_function_call2tc(
    migrate_type(return_type), symbol_expr2tc(fn), args2));
}
} // namespace

std::string python_converter::op_to_dunder(const std::string &op)
{
  static const std::map<std::string, std::string> dunder_map = {
    {"Eq", "__eq__"},
    {"NotEq", "__ne__"},
    {"Lt", "__lt__"},
    {"LtE", "__le__"},
    {"Gt", "__gt__"},
    {"GtE", "__ge__"},
    {"Add", "__add__"},
    {"Sub", "__sub__"},
    {"Mult", "__mul__"},
    {"Div", "__truediv__"},
    {"FloorDiv", "__floordiv__"},
    {"Mod", "__mod__"},
  };
  auto it = dunder_map.find(op);
  return it != dunder_map.end() ? it->second : "";
}

std::string python_converter::op_to_rdunder(const std::string &op)
{
  static const std::map<std::string, std::string> rdunder_map = {
    {"Add", "__radd__"},
    {"Sub", "__rsub__"},
    {"Mult", "__rmul__"},
    {"Div", "__rtruediv__"},
    {"FloorDiv", "__rfloordiv__"},
    {"Mod", "__rmod__"},
  };
  auto it = rdunder_map.find(op);
  return it != rdunder_map.end() ? it->second : "";
}

symbolt *python_converter::find_dunder_method(
  const std::string &class_name,
  const std::string &dunder_name)
{
  std::string tag = "tag-" + class_name;
  const symbolt *type_sym = symbol_table_.find_symbol(tag);
  if (!type_sym)
    return nullptr;

  std::string file = type_sym->location.get_file().as_string();
  if (file.empty())
    return nullptr;

  symbol_id sid(file, class_name, dunder_name);
  return find_symbol(sid.to_string());
}

bool python_converter::has_dunder_method(
  const nlohmann::json &value_node,
  const std::string &dunder_name)
{
  const std::string class_name = type_handler_.get_var_classname(value_node);
  if (class_name.empty())
    return false;

  return find_dunder_method(class_name, dunder_name) != nullptr;
}

nlohmann::json python_converter::build_dunder_call(
  const nlohmann::json &object,
  const std::string &dunder_name,
  const nlohmann::json &args,
  const nlohmann::json &source_node) const
{
  nlohmann::json call_node;
  call_node["_type"] = "Call";
  call_node["func"] = {
    {"_type", "Attribute"}, {"value", object}, {"attr", dunder_name}};
  call_node["args"] = args;
  call_node["keywords"] = nlohmann::json::array();
  if (source_node.contains("lineno"))
    call_node["lineno"] = source_node["lineno"];
  if (source_node.contains("col_offset"))
    call_node["col_offset"] = source_node["col_offset"];
  if (source_node.contains("end_lineno"))
    call_node["end_lineno"] = source_node["end_lineno"];
  if (source_node.contains("end_col_offset"))
    call_node["end_col_offset"] = source_node["end_col_offset"];
  return call_node;
}

static bool is_excluded_struct_tag(const std::string &tag)
{
  return tag.find("dict_") != std::string::npos ||
         tag.find("tag-dict") != std::string::npos ||
         tag.rfind("tag-Optional_", 0) == 0 || tag.rfind("tag-tuple", 0) == 0 ||
         tag == "__python_dict__";
}

static typet resolve_operand_type(
  const exprt &operand,
  const contextt &symbol_table,
  const namespacet &ns)
{
  typet t = operand.type();
  if (operand.is_symbol())
  {
    const symbolt *sym = symbol_table.find_symbol(operand.identifier());
    if (sym)
      t = sym->get_type();
  }
  if (t.id() == "symbol")
    t = ns.follow(t);
  return t;
}

// Check whether the argument type matches the "other" parameter type.
// In case the user annotates it with a concrete class type.
static bool is_other_param_compatible(
  const code_typet &method_type,
  const typet &operand_type,
  const namespacet &ns)
{
  const auto &params = method_type.arguments();
  if (params.size() < 2)
    return true;

  typet param_type = params[1].type();
  if (param_type.id() == "symbol")
    param_type = ns.follow(param_type);

  if (param_type.is_pointer())
  {
    typet subtype = param_type.subtype();
    if (subtype.id() == "symbol")
      subtype = ns.follow(subtype);

    if (subtype.is_struct() && operand_type.is_struct())
      return to_struct_type(subtype).tag() ==
             to_struct_type(operand_type).tag();
  }
  return true;
}

exprt python_converter::dispatch_dunder_operator(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const locationt &loc)
{
  typet lhs_type = resolve_operand_type(lhs, symbol_table_, ns);
  typet rhs_type = resolve_operand_type(rhs, symbol_table_, ns);

  // Try lhs.__add__(rhs)
  if (lhs_type.is_struct())
  {
    const struct_typet &lhs_struct = to_struct_type(lhs_type);
    std::string lhs_tag = lhs_struct.tag().as_string();

    if (!is_excluded_struct_tag(lhs_tag))
    {
      std::string dunder = op_to_dunder(op);
      if (!dunder.empty())
      {
        std::string class_name = extract_class_name_from_tag(lhs_tag);
        symbolt *method = find_dunder_method(class_name, dunder);
        if (method)
        {
          const code_typet &method_type = to_code_type(method->get_type());
          if (is_other_param_compatible(method_type, rhs_type, ns))
          {
            exprt call = build_call_expr(
              *method,
              method_type.return_type(),
              {gen_address_of(lhs), gen_address_of(rhs)});
            call.location() = loc;
            return call;
          }
        }
      }
    }
  }

  // fallback: try rhs.__radd__(lhs)
  if (rhs_type.is_struct())
  {
    const struct_typet &rhs_struct = to_struct_type(rhs_type);
    std::string rhs_tag = rhs_struct.tag().as_string();

    if (!is_excluded_struct_tag(rhs_tag))
    {
      std::string rdunder = op_to_rdunder(op);
      if (!rdunder.empty())
      {
        std::string class_name = extract_class_name_from_tag(rhs_tag);
        symbolt *method = find_dunder_method(class_name, rdunder);
        if (method)
        {
          const code_typet &method_type = to_code_type(method->get_type());
          if (is_other_param_compatible(method_type, lhs_type, ns))
          {
            exprt call = build_call_expr(
              *method,
              method_type.return_type(),
              {gen_address_of(rhs), gen_address_of(lhs)});
            call.location() = loc;
            return call;
          }
        }
      }
    }
  }

  return nil_exprt();
}

exprt python_converter::dispatch_unary_dunder_operator(
  const std::string &op,
  exprt &operand,
  const locationt &loc)
{
  typet operand_type = operand.type();
  if (operand.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(operand.identifier());
    if (sym)
      operand_type = sym->get_type();
  }
  if (operand_type.id() == "symbol")
    operand_type = ns.follow(operand_type);

  if (!operand_type.is_struct())
    return nil_exprt();

  const struct_typet &struct_type = to_struct_type(operand_type);
  std::string tag = struct_type.tag().as_string();

  if (
    tag.find("dict_") != std::string::npos ||
    tag.find("tag-dict") != std::string::npos ||
    tag.rfind("tag-Optional_", 0) == 0 || tag.rfind("tag-tuple", 0) == 0 ||
    tag == "__python_dict__")
    return nil_exprt();

  static const std::map<std::string, std::string> unary_dunder_map = {
    {"USub", "__neg__"},
    {"abs", "__abs__"},
    {"complex", "__complex__"},
    {"float", "__float__"},
    {"index", "__index__"},
    {"str", "__str__"},
  };
  auto it = unary_dunder_map.find(op);
  if (it == unary_dunder_map.end())
    return nil_exprt();

  std::string class_name = extract_class_name_from_tag(tag);
  symbolt *method = find_dunder_method(class_name, it->second);
  if (!method)
    return nil_exprt();

  const code_typet &method_type = to_code_type(method->get_type());
  exprt call = build_call_expr(
    *method, method_type.return_type(), {gen_address_of(operand)});
  call.location() = loc;
  return call;
}
