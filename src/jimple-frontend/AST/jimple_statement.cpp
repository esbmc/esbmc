#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>
#include <util/irep/std_types.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <irep2/irep2_expr.h>
#include <util/arith/arith_tools.h>
#include "util/lang/c_typecast.h"

// Restored: PR #7841 measured this arm unreached over the jimple tests and
// deleted it, but nothing gives the class a native to_code2t, so the default
// reaches it and the base's code_skipt was returned instead. No test builds
// this statement, which is why the deletion was invisible
// (docs/roadmap/scope-jimple-irep2.md §44).
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

void jimple_identity::from_json(const json &j)
{
  j.at("identifier").get_to(at_identifier);
  j.at("name").get_to(local_name);
  j.at("type").get_to(type);
}

std::string jimple_identity::to_string() const
{
  std::ostringstream oss;
  oss << "Identity:  " << this->local_name << " = @" << at_identifier << " | "
      << type.to_string();
  return oss.str();
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

  // Both invoke forms build the call themselves once told where to put the
  // result.
  if (dyn_expr && !dyn_expr->is_nondet_call() && !dyn_expr->is_intrinsic_method)
  {
    dyn_expr->set_lhs2(lhs->to_expr2t(ctx, class_name, function_name));
    return rhs->to_expr2t(ctx, class_name, function_name);
  }

  if (dyn2_expr && !dyn2_expr->is_nondet_call())
  {
    dyn2_expr->set_lhs2(lhs->to_expr2t(ctx, class_name, function_name));
    return rhs->to_expr2t(ctx, class_name, function_name);
  }

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

// Restored: PR #7841 measured this arm unreached over the jimple tests and
// deleted it, but nothing gives the class a native to_code2t, so the default
// reaches it and the base's code_skipt was returned instead. No test builds
// this statement, which is why the deletion was invisible
// (docs/roadmap/scope-jimple-irep2.md §44).
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

  auto symbol = create_jimple_symbolt(
    code_type2tc(
      std::vector<type2tc>{},
      get_empty_type(),
      std::vector<irep_idt>{},
      /*ellipsis=*/false),
    class_name,
    name,
    id,
    function_name);

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

void jimple_assertion::from_json(const json &j)
{
  j.at("equals").at("symbol").get_to(variable);
  j.at("equals").at("value").get_to(value);
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

expr2tc jimple_throw::to_code2t(
  contextt &,
  const std::string &,
  const std::string &,
  const locationt &loc) const
{
  // TODO: throw
  // Since the implementation of Throw isn't complete, neither the thrown
  // operand nor the exception list is populated -- the legacy arm this replaces
  // built a bare codet("cpp-throw") for the same reason.
  return code_cpp_throw2tc(expr2tc(), std::vector<irep_idt>(), loc);
}
