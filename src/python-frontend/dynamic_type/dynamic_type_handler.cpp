#include <python-frontend/dynamic_type/dynamic_type_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type/type_handler.h>
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

  auto then_types = classify_branch_literal_assigns(if_node["body"]);
  if (then_types.empty())
    return dynamic_type_names;

  if (!if_node.contains("orelse") || if_node["orelse"].empty())
    return dynamic_type_names;
  auto else_types = classify_branch_literal_assigns(if_node["orelse"]);

  for (const auto &[name, then_kind] : then_types)
  {
    auto it = else_types.find(name);
    if (it == else_types.end() || it->second == then_kind)
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

void dynamic_type_handler::assign(
  const exprt &rhs,
  const locationt &location,
  const std::string &name,
  codet &target_block)
{
  symbol_id sid = converter_.create_symbol_id();
  sid.set_object(name);
  std::string tag_id = sid.to_string();
  auto alias = aliases_.find(tag_id);
  if (alias != aliases_.end())
    tag_id = alias->second;
  symbolt *tag_symbol = converter_.symbol_table().find_symbol(tag_id);
  assert(
    tag_symbol &&
    "tagged scalar symbol must already be declared before its branches are "
    "converted");

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

exprt dynamic_type_handler::build_eq_literal(
  const exprt &tagged,
  const exprt &literal)
{
  exprt tagged_addr = build_address_of(tagged);
  exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());

  if (literal.type().is_array())
  {
    const symbolt *eq_str_func =
      converter_.symbol_table().find_symbol("c:@F@__python_scalar_eq_str");
    assert(eq_str_func && "__python_scalar_eq_str not found in symbol table");
    exprt lit_addr =
      converter_.get_string_handler().get_array_base_address(literal);
    exprt lit_size = type_handler_.tagged_scalar_byte_size(literal);

    exprt call = build_call_expr(
      *eq_str_func, int_type(), {tagged_addr, lit_type_id, lit_addr, lit_size});
    return build_equal(call, from_integer(1, int_type()));
  }

  if (
    !literal.type().is_bool() && !literal.type().is_signedbv() &&
    !literal.type().is_unsignedbv())
    throw std::runtime_error(
      "comparing a dynamically-typed variable against this literal type is "
      "not yet supported");

  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));

  const symbolt *eq_num_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_eq_num");
  assert(eq_num_func && "__python_scalar_eq_num not found in symbol table");
  exprt call = build_call_expr(
    *eq_num_func, int_type(), {tagged_addr, lit_type_id, lit_value});
  return build_equal(call, from_integer(1, int_type()));
}

exprt dynamic_type_handler::handle_comparison(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  const bool lhs_tagged = type_handler_.is_tagged_scalar_type(lhs.type());
  const bool rhs_tagged = type_handler_.is_tagged_scalar_type(rhs.type());

  // A tagged-vs-tagged compare needs a size known only at runtime (each
  // operand's `.size` field), so the byte-compare's memcmp fallback can't
  // unwind to termination even though the real value is always tiny.
  // Comparing against a literal doesn't have this problem, since its size is
  // a compile-time constant. Refuse cleanly rather than risk a
  // non-terminating run.
  if (lhs_tagged && rhs_tagged)
    throw std::runtime_error(
      "comparing two dynamically-typed variables directly is not yet "
      "supported");

  exprt result =
    lhs_tagged ? build_eq_literal(lhs, rhs) : build_eq_literal(rhs, lhs);

  return op == "NotEq" ? build_not(result) : result;
}

exprt dynamic_type_handler::build_add_literal(
  const exprt &tagged,
  const exprt &literal,
  bool tagged_is_left)
{
  exprt tagged_addr = build_address_of(tagged);
  exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());

  if (literal.type().is_array())
  {
    const symbolt *add_str_func =
      converter_.symbol_table().find_symbol("c:@F@__python_scalar_add_str");
    assert(add_str_func && "__python_scalar_add_str not found in symbol table");
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

  const symbolt *add_num_func =
    converter_.symbol_table().find_symbol("c:@F@__python_scalar_add_num");
  assert(add_num_func && "__python_scalar_add_num not found in symbol table");
  return build_call_expr(
    *add_num_func,
    signedbv_typet(config.ansi_c.long_long_int_width),
    {tagged_addr, lit_type_id, lit_value});
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
  exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());
  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));

  return build_call_expr(
    *sub_func,
    signedbv_typet(config.ansi_c.long_long_int_width),
    {tagged_addr,
     lit_type_id,
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
  exprt lit_type_id = type_handler_.tagged_scalar_type_id(literal.type());
  exprt lit_value =
    build_typecast(literal, signedbv_typet(config.ansi_c.long_long_int_width));

  // A catchable raise, not an assert, so it's guarded only when the type
  // matches.
  exprt tagged_type_id = build_member(tagged, "type_id", size_type());
  exprt type_matches = build_equal(tagged_type_id, lit_type_id);
  exprt tagged_numeric_value = build_dereference(
    build_typecast(
      build_member(tagged, "value", pointer_typet(empty_typet())),
      pointer_typet(signedbv_typet(config.ansi_c.long_long_int_width))),
    signedbv_typet(config.ansi_c.long_long_int_width));
  exprt divisor = tagged_is_left ? lit_value : tagged_numeric_value;
  exprt is_zero = build_equal(divisor, from_integer(BigInt(0), divisor.type()));

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ZeroDivisionError", "division by zero");
  code_expressiont throw_code(raise);

  code_ifthenelset guard;
  guard.cond() = build_and(type_matches, is_zero);
  guard.then_case() = throw_code;
  guard.location() = location;
  converter_.add_instruction(guard);

  return build_call_expr(
    *div_func,
    double_type(),
    {tagged_addr,
     lit_type_id,
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

  // Same runtime-size concern as handle_comparison's tagged-vs-tagged case.
  if (lhs_tagged && rhs_tagged)
    throw std::runtime_error(
      "'" + op +
      "' between two dynamically-typed variables directly is "
      "not yet supported");

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
