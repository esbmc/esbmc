#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>
#include <util/irep/std_types.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <irep2/irep2_expr.h>
#include <util/arith/arith_tools.h>
#include "util/lang/c_typecast.h"

void jimple_identity::from_json(const json &j)
{
  j.at("identifier").get_to(at_identifier);
  j.at("name").get_to(local_name);
  j.at("type").get_to(type);
}

exprt jimple_identity::to_exprt(
  contextt &ctx,
  const std::string &,
  const std::string &) const
{
  // TODO: Symbol-table / Typecast
  exprt val("at_identifier");
  symbolt &added_symbol = *ctx.find_symbol(local_name);
  symbolt rhs;
  rhs.name = "@" + at_identifier;
  rhs.id = "@" + at_identifier;
  code_assignt assign(symbol_expr(added_symbol), symbol_expr(rhs));
  return assign;
}
std::string jimple_identity::to_string() const
{
  std::ostringstream oss;
  oss << "Identity:  " << this->local_name << " = @" << at_identifier << " | "
      << type.to_string();
  return oss.str();
}

exprt jimple_return::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: jimple return with support to other returns
  typet return_type = empty_typet();
  code_returnt ret_expr;
  if (expr)
  {
    auto return_value = expr->to_exprt(ctx, class_name, function_name);
    ret_expr.op0() = return_value;
  }
  // TODO: jimple return should support values
  return ret_expr;
}

expr2tc jimple_return::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name,
  const locationt &loc) const
{
  // code_returnt always carries one operand, nil when there is no value, and
  // migrate_expr maps that nil to a null expr2tc.
  expr2tc value;
  if (expr)
    value = expr->to_expr2t(ctx, class_name, function_name);

  return code_return2tc(value, loc);
}

std::string jimple_return::to_string() const
{
  return "Return: (Nothing)";
}
void jimple_return::from_json(const json &j)
{
  if (j.contains("value"))
    expr = jimple_expr::get_expression(j.at("value"));
}
std::string jimple_label::to_string() const
{
  std::ostringstream oss;
  oss << "Label: " << this->label;
  for (auto member : this->members->members)
    oss << "\n\t\t\t" << member->to_string();
  return oss.str();
}

exprt jimple_label::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: DRY (clang-c-converter)
  code_labelt c_label;
  c_label.set_label(label);

  code_blockt block;
  for (auto member : members->members)
  {
    block.operands().push_back(
      std::move(member->to_exprt(ctx, class_name, function_name)));
  }
  c_label.code() = to_code(block);

  return c_label;
}

// K.3 of docs/roadmap/scope-jimple-irep2.md. migrate_expr's label arm also
// flattens a single-declaration decl-block body to the bare decl; this frontend
// never builds a decl-block, so there is nothing to reproduce. The members are
// passed a nil location because to_exprt above does not stamp them -- only
// jimple_full_method_body does that.
expr2tc jimple_label::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name,
  const locationt &loc) const
{
  const locationt &nil = static_cast<const locationt &>(get_nil_irep());

  std::vector<expr2tc> ops;
  ops.reserve(members->members.size());
  for (auto const &member : members->members)
    ops.push_back(member->to_code2t(ctx, class_name, function_name, nil));

  return code_label2tc(label, code_block2tc(ops, nil, nil), loc);
}

void jimple_goto::from_json(const json &j)
{
  j.at("goto").get_to(label);
}

std::string jimple_goto::to_string() const
{
  std::ostringstream oss;
  oss << "Goto: " << this->label;
  return oss.str();
}

exprt jimple_goto::to_exprt(
  contextt &,
  const std::string &,
  const std::string &) const
{
  code_gotot code_goto;
  code_goto.set_destination(label);
  return code_goto;
}

// K.3 of docs/roadmap/scope-jimple-irep2.md: the first statement to build its
// IREP2 form directly rather than through the base's migrating default. Matches
// migrate_expr's goto arm, which reads the destination off the legacy node's
// "destination" field -- what set_destination writes above.
expr2tc jimple_goto::to_code2t(
  contextt &,
  const std::string &,
  const std::string &,
  const locationt &loc) const
{
  return code_goto2tc(label, loc);
}

void jimple_label::from_json(const json &j)
{
  j.at("label_id").get_to(label);
  jimple_full_method_body b;
  b.from_json(j.at("content"));
  members = std::make_shared<jimple_full_method_body>(b);
}

std::string jimple_assignment::to_string() const
{
  std::ostringstream oss;
  oss << "Assignment: " << lhs->to_string() << " = " << rhs->to_string();
  return oss.str();
}

void jimple_assignment::from_json(const json &j)
{
  lhs = jimple_expr::get_expression(j.at("lhs"));
  rhs = jimple_expr::get_expression(j.at("rhs"));
}

exprt jimple_assignment::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  //TODO: Remove this hack
  if (is_skip)
  {
    code_skipt skip;
    return skip;
  }

  auto lhs_handle = lhs->to_exprt(ctx, class_name, function_name);

  auto dyn_expr = std::dynamic_pointer_cast<jimple_expr_invoke>(rhs);
  if (dyn_expr && !dyn_expr->is_nondet_call() && !dyn_expr->is_intrinsic_method)
  {
    dyn_expr->set_lhs(lhs_handle);
    return rhs->to_exprt(ctx, class_name, function_name);
  }

  auto dyn2_expr = std::dynamic_pointer_cast<jimple_virtual_invoke>(rhs);
  if (dyn2_expr && !dyn2_expr->is_nondet_call())
  {
    dyn2_expr->set_lhs(lhs_handle);
    return rhs->to_exprt(ctx, class_name, function_name);
  }

  auto from_expr = rhs->to_exprt(ctx, class_name, function_name);
  c_typecastt c_typecast(ctx);
  c_typecast.implicit_typecast(from_expr, lhs_handle.type());

  code_assignt assign(lhs_handle, from_expr);
  return assign;
}

expr2tc jimple_assignment::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name,
  const locationt &loc) const
{
  // No is_skip arm to mirror the one in to_exprt: is_skip is initialised false
  // and assigned nowhere in the tree, so that arm is unreachable in both
  // copies. Reproducing it here would be dead instrumentation.

  // Both invoke forms rewrite their own left-hand side and lower to a call
  // rather than to an assignment, so they stay on the migrating default.
  auto dyn_expr = std::dynamic_pointer_cast<jimple_expr_invoke>(rhs);
  auto dyn2_expr = std::dynamic_pointer_cast<jimple_virtual_invoke>(rhs);

  if (
    (dyn_expr && !dyn_expr->is_nondet_call() &&
     !dyn_expr->is_intrinsic_method) ||
    (dyn2_expr && !dyn2_expr->is_nondet_call()))
    return jimple_method_field::to_code2t(ctx, class_name, function_name, loc);

  expr2tc target = lhs->to_expr2t(ctx, class_name, function_name);
  expr2tc source = rhs->to_expr2t(ctx, class_name, function_name);

  // The two c_typecast copies agreed on the conversions jimple can produce
  // only after esbmc/esbmc#6873 aligned the constant fold; jimple_type builds
  // nothing but int, bool, void and pointers, so no other divergence applies
  // (docs/roadmap/scope-coupled-arith-assign-conversion.md §20).
  namespacet ns(ctx);
  c_implicit_typecast(source, target->type, ns);

  return code_assign2tc(target, source, loc);
}

std::string jimple_if::to_string() const
{
  std::ostringstream oss;
  oss << "If: " << cond->to_string() << " THEN GOTO " << label;
  return oss.str();
}

void jimple_if::from_json(const json &j)
{
  cond = jimple_expr::get_expression(j.at("expression"));
  j.at("goto").get_to(label);
}

exprt jimple_if::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_gotot code_goto;
  code_goto.set_destination(label);

  auto condition = cond->to_exprt(ctx, class_name, function_name);
  codet if_expr("ifthenelse");
  if_expr.copy_to_operands(condition, code_goto);

  return if_expr;
}

// The first statement to reach an expression through to_expr2t. migrate_expr's
// ifthenelse arm leaves else_case nil when the legacy node has only two
// operands, which is the shape built above, so the else stays default.
expr2tc jimple_if::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name,
  const locationt &loc) const
{
  expr2tc condition = cond->to_expr2t(ctx, class_name, function_name);
  expr2tc target = code_goto2tc(label);

  return code_ifthenelse2tc(condition, target, expr2tc(), loc);
}

std::string jimple_assertion::to_string() const
{
  std::ostringstream oss;
  oss << "Assertion: " << variable << " = " << value;
  return oss.str();
}

void jimple_assertion::from_json(const json &j)
{
  j.at("equals").at("symbol").get_to(variable);
  j.at("equals").at("value").get_to(value);
}

exprt jimple_assertion::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  code_function_callt call;

  std::ostringstream oss;
  oss << class_name << ":" << function_name << "@" << variable;

  // TODO: move this from here
  std::string id, name;
  id = "__ESBMC_assert";
  name = "__ESBMC_assert";

  auto symbol =
    create_jimple_symbolt(code_typet(), class_name, name, id, function_name);

  symbolt &added_symbol = *ctx.move_symbol_to_context(symbol);

  call.function() = symbol_expr(added_symbol);

  symbolt &test = *ctx.find_symbol(oss.str());
  int as_number = std::stoi(value);
  exprt value_operand = from_integer(as_number, int_type());

  equality_exprt ge(symbol_expr(test), value_operand);
  not_exprt qwe(ge);
  call.arguments().push_back(qwe);

  array_of_exprt arr;
  // TODO: Create binop operation between symbol and value
  return call;
}

std::string jimple_invoke::to_string() const
{
  std::ostringstream oss;
  oss << "Invoke: " << method;
  return oss.str();
}

void jimple_invoke::from_json(const json &j)
{
  j.at("base_class").get_to(base_class);
  j.at("method").get_to(method);
  if (j.contains("variable"))
    j.at("variable").get_to(variable);
  for (auto x : j.at("parameters"))
  {
    parameters.push_back(std::move(jimple_expr::get_expression(x)));
  }
  method += "_" + get_hash_name();
}

exprt jimple_invoke::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: Move intrinsics to backend
  if (base_class == "kotlin.jvm.internal.Intrinsics")
  {
    code_skipt skip;
    return skip;
  }

  // TODO: Move intrinsics to backend
  if (base_class == "java.lang.Runtime")
  {
    code_skipt skip;
    return skip;
  }

  // Don't care for the default object constructor
  if (base_class == "java.lang.Object")
  {
    code_skipt skip;
    return skip;
  }

  // Don't care for Random
  if (base_class == "java.util.Random")
  {
    code_skipt skip;
    return skip;
  }

  // Don't care for String
  if (base_class == "java.lang.String")
  {
    code_skipt skip;
    return skip;
  }

  if (base_class == "java.lang.AssertionError")
  {
    code_skipt skip;
    return skip;
  }

  code_blockt block;
  code_function_callt call;

  std::ostringstream oss;
  oss << base_class << ":" << method;
  auto symbol = ctx.find_symbol(oss.str());
  call.function() = symbol_expr(*symbol);

  if (variable != "")
  {
    // Let's add @THIS
    auto this_expression =
      jimple_symbol(variable).to_exprt(ctx, class_name, function_name);
    call.arguments().push_back(this_expression);
    auto temp = get_symbol_name(base_class, method, "@this");
    symbolt &added_symbol = *ctx.find_symbol(temp);
    code_assignt assign(symbol_expr(added_symbol), this_expression);
    block.operands().push_back(assign);
  }

  for (unsigned long int i = 0; i < parameters.size(); i++)
  {
    // Just adding the arguments should be enough to set the parameters
    auto parameter_expr =
      parameters[i]->to_exprt(ctx, class_name, function_name);
    call.arguments().push_back(parameter_expr);
    // Hack, manually adding parameters
    std::ostringstream oss;
    oss << "@parameter" << i;
    auto temp = get_symbol_name(base_class, method, oss.str());
    symbolt &added_symbol = *ctx.find_symbol(temp);
    code_assignt assign(symbol_expr(added_symbol), parameter_expr);
    block.operands().push_back(assign);
  }
  block.operands().push_back(call);
  return block;
}

expr2tc jimple_invoke::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name,
  const locationt &loc) const
{
  // TODO: Move intrinsics to backend
  static const std::set<std::string> modelled_elsewhere = {
    "kotlin.jvm.internal.Intrinsics",
    "java.lang.Runtime",
    "java.lang.Object",
    "java.util.Random",
    "java.lang.String",
    "java.lang.AssertionError"};

  if (modelled_elsewhere.count(base_class))
    return code_skip2tc(get_empty_type(), loc);

  const locationt &nil = static_cast<const locationt &>(get_nil_irep());

  std::ostringstream oss;
  oss << base_class << ":" << method;
  expr2tc function = symbol_expr2tc(*ctx.find_symbol(oss.str()));

  std::vector<expr2tc> args, ops;

  // The @this / @parameterN assignments mirror to_exprt: the arguments alone
  // do not bind the callee's parameter symbols.
  if (variable != "")
  {
    expr2tc this_expression =
      jimple_symbol(variable).to_expr2t(ctx, class_name, function_name);
    args.push_back(this_expression);
    ops.push_back(code_assign2tc(
      symbol_expr2tc(
        *ctx.find_symbol(get_symbol_name(base_class, method, "@this"))),
      this_expression,
      nil));
  }

  for (std::size_t i = 0; i < parameters.size(); i++)
  {
    expr2tc parameter_expr =
      parameters[i]->to_expr2t(ctx, class_name, function_name);
    args.push_back(parameter_expr);

    std::ostringstream parameter_name;
    parameter_name << "@parameter" << i;
    ops.push_back(code_assign2tc(
      symbol_expr2tc(*ctx.find_symbol(
        get_symbol_name(base_class, method, parameter_name.str()))),
      parameter_expr,
      nil));
  }

  ops.push_back(code_function_call2tc(expr2tc(), function, args, nil));

  return code_block2tc(ops, loc, nil);
}

std::string jimple_throw::to_string() const
{
  std::ostringstream oss;
  oss << "Throw: " << expr->to_string();
  return oss.str();
}

void jimple_throw::from_json(const json &j)
{
  expr = jimple_expr::get_expression(j.at("expr"));
}

exprt jimple_throw::to_exprt(
  contextt &,
  const std::string &,
  const std::string &) const
{
  codet p = codet("cpp-throw");
  // TODO: throw
  // Since the implementation of Throw isn't complete,
  // the expression shouldn't be used.

  // auto to_add = expr->to_exprt(ctx, class_name, function_name);
  // p.move_to_operands(to_add);
  return p;
}
