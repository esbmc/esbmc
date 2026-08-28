#include <python-frontend/dynamic_type/dynamic_type_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type/type_handler.h>
#include <python-frontend/type/type_utils.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <util/arith/arith_tools.h>
#include <util/config/config.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>
#include <util/lang/c_types.h>

using namespace python_expr;

namespace
{
// Classifies a branch's direct, top-level literal assignments by literal
// type.
std::unordered_map<std::string, std::string>
classify_branch_literal_assigns(const nlohmann::json &block)
{
  std::unordered_map<std::string, std::string> types;
  if (!block.is_array())
    return types;

  for (const auto &stmt : block)
  {
    if (!stmt.is_object())
      continue;

    const std::string stmt_type = stmt.value("_type", "");
    nlohmann::json target;
    if (stmt_type == "Assign")
    {
      if (!stmt.contains("targets") || stmt["targets"].size() != 1)
        continue;
      target = stmt["targets"][0];
    }
    else if (stmt_type == "AnnAssign")
    {
      if (!stmt.contains("target"))
        continue;
      target = stmt["target"];
    }
    else
      continue;

    if (target.value("_type", "") != "Name" || !target.contains("id"))
      continue;
    const std::string &name = target["id"].get<std::string>();

    // A later reassignment invalidates any literal kind recorded for `name`
    // by an earlier statement in this same block.
    if (!stmt.contains("value") || stmt["value"].is_null())
    {
      types.erase(name);
      continue;
    }
    const auto &value = stmt["value"];
    if (value.value("_type", "") != "Constant" || !value.contains("value"))
    {
      types.erase(name);
      continue;
    }

    const auto &lit = value["value"];
    if (lit.is_string())
      types[name] = "str";
    else if (lit.is_number_integer() || lit.is_boolean())
      types[name] = "num";
    else
      types.erase(name);
  }

  return types;
}

// Recursively collects, per name, the literal kinds assigned to it across
// every leaf reachable from `block` -- following a nested If (an elif in
// orelse, or a plain nested if/else in body) into both of its arms.
// Returns false if any such nested If is dangling (no final else).
bool collect_branch_literal_kinds(
  const nlohmann::json &block,
  std::unordered_map<std::string, std::unordered_set<std::string>> &kinds,
  std::unordered_map<std::string, int> &leaf_count,
  int &leaf_total)
{
  if (
    block.is_array() && block.size() == 1 && block[0].is_object() &&
    block[0].value("_type", "") == "If")
  {
    const auto &nested = block[0];
    if (
      !nested.contains("body") || !nested.contains("orelse") ||
      nested["orelse"].empty())
      return false;

    return collect_branch_literal_kinds(
             nested["body"], kinds, leaf_count, leaf_total) &&
           collect_branch_literal_kinds(
             nested["orelse"], kinds, leaf_count, leaf_total);
  }

  leaf_total++;
  for (const auto &[name, kind] : classify_branch_literal_assigns(block))
  {
    kinds[name].insert(kind);
    leaf_count[name]++;
  }
  return true;
}

// Classifies a single `return <literal>` statement: "num", "str", or "" if
// it isn't a literal return.
std::string classify_return_literal_kind(const nlohmann::json &ret_stmt)
{
  if (!ret_stmt.is_object() || ret_stmt.value("_type", "") != "Return")
    return "";
  if (!ret_stmt.contains("value") || ret_stmt["value"].is_null())
    return "";

  const auto &value = ret_stmt["value"];
  if (value.value("_type", "") != "Constant" || !value.contains("value"))
    return "";

  const auto &lit = value["value"];
  if (lit.is_string())
    return "str";
  if (lit.is_number_integer() || lit.is_boolean())
    return "num";
  return "";
}

// Collects every literal return kind reachable from the tail of `block`,
// following elif chains (a nested If as the block's last statement) into
// both of their arms.
void collect_branch_return_kinds(
  const nlohmann::json &block,
  std::unordered_set<std::string> &kinds)
{
  if (!block.is_array() || block.empty())
    return;

  const auto &last = block.back();
  if (last.is_object() && last.value("_type", "") == "If")
  {
    if (last.contains("body"))
      collect_branch_return_kinds(last["body"], kinds);
    if (last.contains("orelse"))
      collect_branch_return_kinds(last["orelse"], kinds);
    return;
  }

  const std::string kind = classify_return_literal_kind(last);
  if (!kind.empty())
    kinds.insert(kind);
}
} // namespace

dynamic_type_handler::dynamic_type_handler(
  python_converter &converter,
  type_handler &type_handler)
  : converter_(converter), type_handler_(type_handler)
{
}

std::unordered_set<std::string> dynamic_type_handler::detect_dynamic_type_names(
  const nlohmann::json &if_node) const
{
  std::unordered_set<std::string> dynamic_type_names;

  if (!if_node.contains("body"))
    return dynamic_type_names;
  if (!if_node.contains("orelse") || if_node["orelse"].empty())
    return dynamic_type_names;

  std::unordered_map<std::string, std::unordered_set<std::string>> kinds;
  std::unordered_map<std::string, int> leaf_count;
  int leaf_total = 0;

  if (!collect_branch_literal_kinds(
        if_node["body"], kinds, leaf_count, leaf_total))
    return dynamic_type_names;
  if (!collect_branch_literal_kinds(
        if_node["orelse"], kinds, leaf_count, leaf_total))
    return dynamic_type_names;

  for (const auto &[name, kind_set] : kinds)
  {
    if (leaf_count[name] != leaf_total || kind_set.size() < 2)
      continue;

    symbol_id sid = converter_.create_symbol_id();
    sid.set_object(name);
    const symbolt *existing =
      converter_.symbol_table().find_symbol(sid.to_string());
    if (
      existing && !type_handler_.is_numeric_scalar_type(existing->get_type()) &&
      !type_handler_.is_string_type(existing->get_type()))
      continue;

    dynamic_type_names.insert(name);
  }

  return dynamic_type_names;
}

void dynamic_type_handler::declare_dynamic_type_names(
  const std::unordered_set<std::string> &dynamic_type_names,
  const nlohmann::json &ast_node)
{
  for (const auto &name : dynamic_type_names)
  {
    symbol_id tag_sid = converter_.create_symbol_id();
    tag_sid.set_object(name);

    // A pre-existing variable already has GOTO instructions carrying its old
    // (non-struct) type; retyping that symbol in place would leave those
    // instructions inconsistent with the symbol table, which whole-program
    // pointer analysis assumes agree. Create a new struct-typed symbol
    // instead and alias the name to it, mirroring the straight-line retype
    // heuristic, but without ever reverting the alias, since surviving the
    // join is the point.
    std::string tag_id = tag_sid.to_string();
    symbolt *existing_symbol = converter_.symbol_table().find_symbol(tag_id);
    if (existing_symbol != nullptr)
    {
      std::string new_id;
      unsigned gen = 1;
      do
      {
        new_id = tag_id + "$tag" + std::to_string(gen++);
      } while (converter_.symbol_table().find_symbol(new_id) != nullptr);

      locationt alias_location = converter_.get_location_from_decl(ast_node);
      std::string alias_module_name = alias_location.get_file().as_string();
      symbolt alias_symbol = converter_.create_symbol(
        alias_module_name,
        name,
        new_id,
        alias_location,
        type_handler_.get_tagged_object_type());
      alias_symbol.lvalue = true;
      alias_symbol.file_local = !converter_.get_current_func_name().empty();
      alias_symbol.is_extern = false;

      symbolt *alias_symbol_ptr =
        converter_.symbol_table().move_symbol_to_context(alias_symbol);
      aliases_[tag_id] = new_id;

      if (
        !converter_.get_current_func_name().empty() &&
        !converter_.is_global_variable(tag_sid))
      {
        code_declt alias_decl(build_symbol(*alias_symbol_ptr));
        alias_decl.location() = alias_location;
        converter_.add_instruction(alias_decl);
      }

      // Initialize from the pre-existing value, in case something reads the
      // alias before a branch's assign() runs.
      exprt existing_value = build_symbol(*existing_symbol);
      for (const codet &instr : build_tag_field_assigns(
             *alias_symbol_ptr, existing_value, alias_location))
        converter_.add_instruction(instr);
      continue;
    }

    locationt tag_location = converter_.get_location_from_decl(ast_node);
    std::string tag_module_name = tag_location.get_file().as_string();
    symbolt tag_symbol = converter_.create_symbol(
      tag_module_name,
      name,
      tag_sid.to_string(),
      tag_location,
      type_handler_.get_tagged_object_type());
    tag_symbol.lvalue = true;
    tag_symbol.file_local = !converter_.get_current_func_name().empty();
    tag_symbol.is_extern = false;

    symbolt *tag_symbol_ptr =
      converter_.symbol_table().move_symbol_to_context(tag_symbol);

    if (
      !converter_.get_current_func_name().empty() &&
      !converter_.is_global_variable(tag_sid))
    {
      code_declt tag_decl(build_symbol(*tag_symbol_ptr));
      tag_decl.location() = tag_location;
      converter_.add_instruction(tag_decl);
    }
  }
}

bool dynamic_type_handler::is_tagged(const std::string &name) const
{
  if (tagged_names_.count(name) > 0)
    return true;

  symbol_id sid = converter_.create_symbol_id();
  sid.set_object(name);
  const std::string tag_id = sid.to_string();
  if (aliases_.count(tag_id) > 0)
    return true;

  const symbolt *sym = converter_.symbol_table().find_symbol(tag_id);
  return sym && type_handler_.is_tagged_scalar_type(sym->get_type());
}

std::string
dynamic_type_handler::tagged_symbol_id(const std::string &name) const
{
  symbol_id sid = converter_.create_symbol_id();
  sid.set_object(name);
  const std::string tag_id = sid.to_string();
  auto alias = aliases_.find(tag_id);
  return alias != aliases_.end() ? alias->second : tag_id;
}

bool dynamic_type_handler::tagged_binop_result_may_be_tagged(
  const std::string &op) const
{
  return op == "Add";
}

std::vector<codet> dynamic_type_handler::build_tag_field_assigns(
  symbolt &tag_symbol,
  const exprt &rhs,
  const locationt &location)
{
  std::vector<codet> instructions;

  exprt value_to_store = rhs;
  if (rhs.type() == bool_type())
    value_to_store =
      build_typecast(rhs, signedbv_typet(config.ansi_c.long_long_int_width));

  symbolt &backing = converter_.create_tmp_symbol(
    location, "$scalar_tag$", value_to_store.type(), value_to_store);
  code_declt backing_decl(build_symbol(backing));
  backing_decl.copy_to_operands(value_to_store);
  backing_decl.location() = location;
  instructions.push_back(backing_decl);

  exprt elem_size = type_handler_.tagged_scalar_byte_size(value_to_store);

  // `backing` goes DEAD at the end of this branch; copy its value into
  // non-expiring storage first so `.value` stays valid past the join.
  const symbolt *copy_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_tag_copy");

  assert(copy_func && "__python_scalar_tag_copy not found in symbol table");

  exprt backing_addr = build_typecast(
    build_address_of(build_symbol(backing)), pointer_typet(empty_typet()));
  exprt copy_call = build_call_expr(
    *copy_func, pointer_typet(empty_typet()), {backing_addr, elem_size});

  exprt type_id_value = type_handler_.tagged_scalar_type_id(rhs.type());

  exprt tag_expr = build_symbol(tag_symbol);

  code_assignt value_assign(
    build_member(tag_expr, "value", pointer_typet(empty_typet())), copy_call);
  value_assign.location() = location;
  instructions.push_back(value_assign);

  code_assignt type_id_assign(
    build_member(tag_expr, "type_id", size_type()), type_id_value);
  type_id_assign.location() = location;
  instructions.push_back(type_id_assign);

  code_assignt size_assign(
    build_member(tag_expr, "size", size_type()), elem_size);
  size_assign.location() = location;
  instructions.push_back(size_assign);

  code_assignt float_idx_assign(
    build_member(tag_expr, "float_idx", size_type()),
    from_integer(BigInt(0), size_type()));
  float_idx_assign.location() = location;
  instructions.push_back(float_idx_assign);

  return instructions;
}

symbolt *dynamic_type_handler::find_tag_symbol(const std::string &name) const
{
  symbolt *tag_symbol =
    converter_.symbol_table().find_symbol(tagged_symbol_id(name));
  assert(
    tag_symbol &&
    "tagged scalar symbol must already be declared before its branches are "
    "converted");
  return tag_symbol;
}

void dynamic_type_handler::assign(
  const exprt &rhs,
  const locationt &location,
  const std::string &name,
  codet &target_block)
{
  symbolt *tag_symbol = find_tag_symbol(name);

  for (const codet &instr : build_tag_field_assigns(*tag_symbol, rhs, location))
    target_block.copy_to_operands(instr);
}

void dynamic_type_handler::resolve_read(symbolt *&symbol) const
{
  auto alias = aliases_.find(symbol->id.as_string());
  if (alias != aliases_.end())
  {
    if (symbolt *tagged = converter_.symbol_table().find_symbol(alias->second))
      symbol = tagged;
  }
}

exprt dynamic_type_handler::build_literal_compare_call(
  const std::string &num_func_name,
  const std::string &str_func_name,
  const exprt &tagged,
  const exprt &literal)
{
  exprt tagged_addr = build_address_of(tagged);

  if (literal.type().is_array())
  {
    const symbolt *str_func =
      converter_.symbol_table().find_symbol("c:@F@" + str_func_name);
    assert(
      str_func && "tagged-vs-literal str helper not found in symbol table");
    exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());
    exprt lit_addr =
      converter_.get_string_handler().get_array_base_address(literal);
    exprt lit_size = type_handler_.tagged_scalar_byte_size(literal);

    return build_call_expr(
      *str_func, int_type(), {tagged_addr, lit_type_id, lit_addr, lit_size});
  }

  if (
    !literal.type().is_bool() && !literal.type().is_signedbv() &&
    !literal.type().is_unsignedbv())
    throw std::runtime_error(
      "comparing a dynamically-typed variable against this literal type is "
      "not yet supported");

  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());
  exprt type_matches = build_typecast(
    type_handler_.tagged_scalar_type_matches(tagged_type_id, literal.type()),
    int_type());

  const symbolt *num_func =
    converter_.symbol_table().find_symbol("c:@F@" + num_func_name);
  assert(num_func && "tagged-vs-literal num helper not found in symbol table");
  return build_call_expr(
    *num_func, int_type(), {tagged_addr, type_matches, lit_value});
}

exprt dynamic_type_handler::build_eq_literal(
  const exprt &tagged,
  const exprt &literal)
{
  exprt call = build_literal_compare_call(
    "__python_scalar_eq_num", "__python_scalar_eq_str", tagged, literal);
  return build_equal(call, from_integer(1, int_type()));
}

exprt dynamic_type_handler::handle_comparison(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  const bool lhs_tagged = type_handler_.is_tagged_scalar_type(lhs.type());
  const bool rhs_tagged = type_handler_.is_tagged_scalar_type(rhs.type());
  const bool ordered = type_utils::is_ordered_comparison(op);

  if (lhs_tagged && rhs_tagged)
  {
    if (ordered)
      throw std::runtime_error(
        "ordering two dynamically-typed variables directly is not yet "
        "supported");

    const symbolt *eq_obj_func =
      converter_.symbol_table().find_symbol("c:@F@__python_scalar_eq_obj");
    assert(eq_obj_func && "__python_scalar_eq_obj not found in symbol table");
    exprt call = build_call_expr(
      *eq_obj_func,
      int_type(),
      {build_address_of(lhs),
       build_address_of(rhs),
       type_handler_.tagged_scalar_type_id(long_long_int_type())});
    exprt equal = build_equal(call, from_integer(1, int_type()));
    return op == "NotEq" ? build_not(equal) : equal;
  }

  if (ordered)
  {
    // Comparison isn't symmetric like ==, so a literal on the left (e.g.
    // `5 < x`) needs the operator mirrored before dispatch.
    static const std::unordered_map<std::string, std::string> mirrored_op = {
      {"Lt", "Gt"}, {"Gt", "Lt"}, {"LtE", "GtE"}, {"GtE", "LtE"}};
    const std::string effective_op = lhs_tagged ? op : mirrored_op.at(op);
    return lhs_tagged ? build_ordered_literal(effective_op, lhs, rhs)
                      : build_ordered_literal(effective_op, rhs, lhs);
  }

  exprt result =
    lhs_tagged ? build_eq_literal(lhs, rhs) : build_eq_literal(rhs, lhs);

  return op == "NotEq" ? build_not(result) : result;
}

exprt dynamic_type_handler::build_ordered_literal(
  const std::string &op,
  const exprt &tagged,
  const exprt &literal)
{
  exprt cmp = build_literal_compare_call(
    "__python_scalar_cmp_num", "__python_scalar_cmp_str", tagged, literal);

  exprt zero = from_integer(BigInt(0), int_type());
  if (op == "Lt")
    return build_less_than(cmp, zero);
  if (op == "LtE")
    return build_less_equal(cmp, zero);
  if (op == "Gt")
    return build_greater_than(cmp, zero);
  assert(op == "GtE" && "unexpected operator routed to build_ordered_literal");
  return build_greater_equal(cmp, zero);
}

exprt dynamic_type_handler::build_add_literal(
  const exprt &tagged,
  const exprt &literal,
  bool tagged_is_left)
{
  exprt tagged_addr = build_address_of(tagged);

  if (literal.type().is_array())
  {
    const symbolt *add_str_func =
      converter_.symbol_table().find_symbol("c:@F@__python_scalar_add_str");
    assert(add_str_func && "__python_scalar_add_str not found in symbol table");
    exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());
    exprt lit_addr =
      converter_.get_string_handler().get_array_base_address(literal);
    exprt lit_size = type_handler_.tagged_scalar_byte_size(literal);

    return build_call_expr(
      *add_str_func,
      pointer_typet(char_type()),
      {tagged_addr,
       lit_type_id,
       lit_addr,
       lit_size,
       from_integer(BigInt(tagged_is_left ? 1 : 0), int_type())});
  }

  if (
    !literal.type().is_bool() && !literal.type().is_signedbv() &&
    !literal.type().is_unsignedbv())
    throw std::runtime_error(
      "adding a dynamically-typed variable to this literal type is not yet "
      "supported");

  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());
  exprt type_matches = build_typecast(
    type_handler_.tagged_scalar_type_matches(tagged_type_id, literal.type()),
    int_type());

  const symbolt *add_num_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_add_num");
  assert(add_num_func && "__python_scalar_add_num not found in symbol table");
  return build_call_expr(
    *add_num_func,
    signedbv_typet(config.ansi_c.long_long_int_width),
    {tagged_addr, type_matches, lit_value});
}

exprt dynamic_type_handler::build_sub_literal(
  const exprt &tagged,
  const exprt &literal,
  bool tagged_is_left)
{
  if (
    !literal.type().is_bool() && !literal.type().is_signedbv() &&
    !literal.type().is_unsignedbv())
    throw std::runtime_error(
      "subtracting a dynamically-typed variable and this literal type is "
      "not yet supported");

  const symbolt *sub_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_sub_num");
  assert(sub_func && "__python_scalar_sub_num not found in symbol table");

  exprt tagged_addr = build_address_of(tagged);
  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());
  exprt type_matches = build_typecast(
    type_handler_.tagged_scalar_type_matches(tagged_type_id, literal.type()),
    int_type());

  return build_call_expr(
    *sub_func,
    signedbv_typet(config.ansi_c.long_long_int_width),
    {tagged_addr,
     type_matches,
     lit_value,
     from_integer(BigInt(tagged_is_left ? 1 : 0), int_type())});
}

exprt dynamic_type_handler::build_div_literal(
  const exprt &tagged,
  const exprt &literal,
  bool tagged_is_left,
  const locationt &location)
{
  if (
    !literal.type().is_bool() && !literal.type().is_signedbv() &&
    !literal.type().is_unsignedbv())
    throw std::runtime_error(
      "dividing a dynamically-typed variable by this literal type is not "
      "yet supported");

  const symbolt *div_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_div_num");
  assert(div_func && "__python_scalar_div_num not found in symbol table");

  exprt tagged_addr = build_address_of(tagged);
  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));

  // A catchable raise, not an assert, so it's guarded only when the type
  // matches.
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());
  exprt type_matches =
    type_handler_.tagged_scalar_type_matches(tagged_type_id, literal.type());
  exprt tagged_numeric_value = build_dereference(
    build_typecast(
      build_member(tagged, "value", pointer_typet(empty_typet())),
      pointer_typet(signedbv_typet(config.ansi_c.long_long_int_width))),
    signedbv_typet(config.ansi_c.long_long_int_width));
  exprt divisor = tagged_is_left ? lit_value : tagged_numeric_value;
  guard_zero_division(divisor, type_matches, location);

  return build_call_expr(
    *div_func,
    double_type(),
    {tagged_addr,
     build_typecast(type_matches, int_type()),
     lit_value,
     from_integer(BigInt(tagged_is_left ? 1 : 0), int_type())});
}

exprt dynamic_type_handler::handle_arithmetic(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs,
  const locationt &location)
{
  const bool lhs_tagged = type_handler_.is_tagged_scalar_type(lhs.type());
  const bool rhs_tagged = type_handler_.is_tagged_scalar_type(rhs.type());

  // Equality can settle a type_id mismatch without knowing either type;
  // arithmetic cannot, since it needs the concrete type to pick an operation.
  if (lhs_tagged && rhs_tagged)
  {
    if (op == "Add")
      return build_add_tagged(lhs, rhs);
    if (op == "Sub")
      return build_sub_tagged(lhs, rhs);
    assert(op == "Div" && "unexpected operator routed to handle_arithmetic");
    return build_div_tagged(lhs, rhs, location);
  }

  if (op == "Add")
    return lhs_tagged ? build_add_literal(lhs, rhs, /*tagged_is_left=*/true)
                      : build_add_literal(rhs, lhs, /*tagged_is_left=*/false);

  if (op == "Sub")
    return lhs_tagged ? build_sub_literal(lhs, rhs, /*tagged_is_left=*/true)
                      : build_sub_literal(rhs, lhs, /*tagged_is_left=*/false);

  assert(op == "Div" && "unexpected operator routed to handle_arithmetic");
  return lhs_tagged ? build_div_literal(lhs, rhs, true, location)
                    : build_div_literal(rhs, lhs, false, location);
}

exprt dynamic_type_handler::build_add_tagged(const exprt &lhs, const exprt &rhs)
{
  const symbolt *add_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_add_obj_dyn");
  assert(add_func && "__python_scalar_add_obj_dyn not found in symbol table");

  exprt lhs_type_id = build_member(lhs, "type_id", size_type());
  exprt rhs_type_id = build_member(rhs, "type_id", size_type());
  exprt str_type_id =
    type_handler_.tagged_scalar_type_id(pointer_typet(char_type()));
  exprt lhs_is_num = build_typecast(
    type_handler_.tagged_scalar_type_matches(lhs_type_id, long_long_int_type()),
    int_type());
  exprt rhs_is_num = build_typecast(
    type_handler_.tagged_scalar_type_matches(rhs_type_id, long_long_int_type()),
    int_type());
  exprt lhs_is_str =
    build_typecast(build_equal(lhs_type_id, str_type_id), int_type());
  exprt rhs_is_str =
    build_typecast(build_equal(rhs_type_id, str_type_id), int_type());

  return build_call_expr(
    *add_func,
    type_handler_.get_tagged_object_type(),
    {build_address_of(lhs),
     lhs_is_num,
     lhs_is_str,
     build_address_of(rhs),
     rhs_is_num,
     rhs_is_str,
     type_handler_.tagged_scalar_type_id(long_long_int_type()),
     str_type_id});
}

exprt dynamic_type_handler::build_sub_tagged(const exprt &lhs, const exprt &rhs)
{
  const symbolt *sub_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_sub_num_dyn");
  assert(sub_func && "__python_scalar_sub_num_dyn not found in symbol table");

  exprt lhs_type_id = build_member(lhs, "type_id", size_type());
  exprt rhs_type_id = build_member(rhs, "type_id", size_type());
  exprt lhs_is_num = build_typecast(
    type_handler_.tagged_scalar_type_matches(lhs_type_id, long_long_int_type()),
    int_type());
  exprt rhs_is_num = build_typecast(
    type_handler_.tagged_scalar_type_matches(rhs_type_id, long_long_int_type()),
    int_type());

  return build_call_expr(
    *sub_func,
    signedbv_typet(config.ansi_c.long_long_int_width),
    {build_address_of(lhs), lhs_is_num, build_address_of(rhs), rhs_is_num});
}

exprt dynamic_type_handler::build_div_tagged(
  const exprt &lhs,
  const exprt &rhs,
  const locationt &location)
{
  const symbolt *div_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_div_num_dyn");
  assert(div_func && "__python_scalar_div_num_dyn not found in symbol table");

  exprt lhs_type_id = build_member(lhs, "type_id", size_type());
  exprt rhs_type_id = build_member(rhs, "type_id", size_type());
  exprt lhs_is_num =
    type_handler_.tagged_scalar_type_matches(lhs_type_id, long_long_int_type());
  exprt rhs_is_num =
    type_handler_.tagged_scalar_type_matches(rhs_type_id, long_long_int_type());

  exprt rhs_numeric_value = build_dereference(
    build_typecast(
      build_member(rhs, "value", pointer_typet(empty_typet())),
      pointer_typet(signedbv_typet(config.ansi_c.long_long_int_width))),
    signedbv_typet(config.ansi_c.long_long_int_width));
  guard_zero_division(
    rhs_numeric_value, build_and(lhs_is_num, rhs_is_num), location);

  return build_call_expr(
    *div_func,
    double_type(),
    {build_address_of(lhs),
     build_typecast(lhs_is_num, int_type()),
     build_address_of(rhs),
     build_typecast(rhs_is_num, int_type())});
}

void dynamic_type_handler::guard_zero_division(
  const exprt &divisor,
  const exprt &type_ok,
  const locationt &location)
{
  exprt is_zero = build_equal(divisor, from_integer(BigInt(0), divisor.type()));

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ZeroDivisionError", "division by zero");
  code_expressiont throw_code(raise);

  code_ifthenelset guard;
  guard.cond() = build_and(type_ok, is_zero);
  guard.then_case() = throw_code;
  guard.location() = location;
  guard.location().property("skipped");
  converter_.add_instruction(guard);
}

exprt dynamic_type_handler::build_isinstance_check(
  const exprt &tagged,
  const std::string &type_name,
  bool type_is_user_class) const
{
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());

  if (type_name == "bool")
    return build_equal(
      tagged_type_id, type_handler_.tagged_scalar_type_id(bool_type()));

  if (type_name == "int")
    return type_handler_.tagged_scalar_type_matches(
      tagged_type_id, long_long_int_type());

  if (type_name == "float")
    return build_equal(
      tagged_type_id, type_handler_.tagged_scalar_type_id(double_type()));

  if (type_name == "str")
    return build_equal(
      tagged_type_id,
      type_handler_.tagged_scalar_type_id(pointer_typet(char_type())));

  // object is Python's top type; every value is an instance of it.
  if (type_name == "object")
    return true_exprt();

  // A tag only ever holds a bool, int, float or str -- get_var_assign refuses
  // every other rvalue -- so no aggregate or user class can ever match (#7075).
  static const std::unordered_set<std::string> unholdable = {
    "NoneType",
    "bytearray",
    "bytes",
    "complex",
    "dict",
    "frozenset",
    "list",
    "range",
    "set",
    "tuple",
    "type"};
  if (type_is_user_class || unholdable.count(type_name))
    return false_exprt();

  throw std::runtime_error(
    "isinstance() against this type is not yet supported for a "
    "dynamically-typed variable");
}

bool dynamic_type_handler::detect_dynamic_return_type(
  const nlohmann::json &function_body) const
{
  if (!function_body.is_array())
    return false;

  for (size_t i = 0; i < function_body.size(); ++i)
  {
    const auto &stmt = function_body[i];
    if (!stmt.is_object() || stmt.value("_type", "") != "If")
      continue;
    if (!stmt.contains("body") || stmt["body"].empty())
      continue;

    std::unordered_set<std::string> kinds;
    collect_branch_return_kinds(stmt["body"], kinds);

    if (stmt.contains("orelse") && !stmt["orelse"].empty())
      collect_branch_return_kinds(stmt["orelse"], kinds);
    else if (i + 1 < function_body.size())
    {
      // No else: an idiomatic early return falls through to the next
      // statement at this level.
      const std::string next_kind =
        classify_return_literal_kind(function_body[i + 1]);
      if (!next_kind.empty())
        kinds.insert(next_kind);
    }

    if (kinds.size() > 1)
      return true;
  }

  return false;
}

exprt dynamic_type_handler::build_tagged_return_value(
  const exprt &value,
  const locationt &location,
  codet &target_block)
{
  symbolt &tag_symbol = converter_.create_tmp_symbol(
    location, "$return_tag$", type_handler_.get_tagged_object_type(), exprt());

  code_declt tag_decl(build_symbol(tag_symbol));
  tag_decl.location() = location;
  target_block.copy_to_operands(tag_decl);

  for (const codet &instr :
       build_tag_field_assigns(tag_symbol, value, location))
    target_block.copy_to_operands(instr);

  return build_symbol(tag_symbol);
}

void dynamic_type_handler::assign_tagged_object(
  const exprt &rhs,
  const locationt &location,
  const std::string &name,
  codet &target_block)
{
  symbolt *tag_symbol = find_tag_symbol(name);

  code_assignt whole_struct_copy(build_symbol(*tag_symbol), rhs);
  whole_struct_copy.location() = location;
  target_block.copy_to_operands(whole_struct_copy);
}

dynamic_type_handler::scope_guard::scope_guard(
  dynamic_type_handler &handler,
  const std::unordered_set<std::string> &dynamic_type_names)
  : handler_(handler)
{
  for (const auto &c : dynamic_type_names)
    if (handler_.tagged_names_.insert(c).second)
      added_.insert(c);
}

dynamic_type_handler::scope_guard::~scope_guard()
{
  for (const auto &a : added_)
    handler_.tagged_names_.erase(a);
}
