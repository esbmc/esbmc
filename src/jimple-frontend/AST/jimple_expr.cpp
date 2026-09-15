#include <jimple-frontend/AST/jimple_expr.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>

void jimple_constant::from_json(const json &j)
{
  j.at("value").get_to(value);
}

exprt jimple_constant::to_exprt(
  contextt &,
  const std::string &,
  const std::string &) const
{
  auto as_number = std::stoi(value);
  return constant_exprt(
    integer2binary(as_number, 10), integer2string(as_number), int_type());
};

// The leaf of this frontend's expression tree: a literal with no context and no
// operands, so it converts with nothing left to migrate. Matches what
// migrate_expr makes of the constant_exprt above -- int_type() is signedbv, so
// the IREP2 form is a constant_int2t of the same width.
expr2tc jimple_constant::to_expr2t(
  contextt &,
  const std::string &,
  const std::string &) const
{
  return constant_int2tc(migrate_type(int_type()), BigInt(std::stoi(value)));
}

void jimple_symbol::from_json(const json &j)
{
  j.at("value").get_to(var_name);
}

exprt jimple_symbol::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // 1. Look over the local scope
  auto symbol_name = get_symbol_name(class_name, function_name, var_name);
  symbolt &s = *ctx.find_symbol(symbol_name);

  // TODO:
  // 2. Look over the class scope
  // 3. Look over the global scope (possibly don't need)

  return symbol_expr(s);
};

expr2tc jimple_symbol::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  return symbol_expr2tc(
    *ctx.find_symbol(get_symbol_name(class_name, function_name, var_name)));
}

std::shared_ptr<jimple_expr> jimple_expr::get_expression(const json &j)
{
  std::string expr_type;
  if (!j.contains("expr_type"))
  {
    jimple_constant c;
    c.setValue("0");
    return std::make_shared<jimple_constant>(c);
  }

  j.at("expr_type").get_to(expr_type);

  // TODO: hashmap, the standard is not stable enough yet
  // It is still a work in progress in the parser: https://github.com/rafaelsamenezes/jimple_parser
  if (expr_type == "constant")
  {
    jimple_constant c;
    c.from_json(j);
    return std::make_shared<jimple_constant>(c);
  }

  if (expr_type == "string_constant")
  {
    jimple_constant c;
    return std::make_shared<jimple_constant>(c);
  }

  if (expr_type == "class_reference")
  {
    jimple_constant c("-1");
    return std::make_shared<jimple_constant>(c);
  }

  if (expr_type == "symbol")
  {
    jimple_symbol c;
    c.from_json(j);
    return std::make_shared<jimple_symbol>(c);
  }

  if (expr_type == "static_invoke")
  {
    jimple_expr_invoke c;
    c.from_json(j);
    return std::make_shared<jimple_expr_invoke>(c);
  }

  if (expr_type == "virtual_invoke")
  {
    jimple_virtual_invoke c;
    c.from_json(j);
    return std::make_shared<jimple_virtual_invoke>(c);
  }

  if (expr_type == "binop")
  {
    jimple_binop c;
    c.from_json(j);
    return std::make_shared<jimple_binop>(c);
  }

  if (expr_type == "cast")
  {
    jimple_cast c;
    c.from_json(j);
    return std::make_shared<jimple_cast>(c);
  }

  if (expr_type == "lengthof")
  {
    jimple_lengthof c;
    c.from_json(j);
    return std::make_shared<jimple_lengthof>(c);
  }

  if (expr_type == "newarray")
  {
    jimple_newarray c;
    c.from_json(j);
    return std::make_shared<jimple_newarray>(c);
  }

  if (expr_type == "new")
  {
    jimple_new c;
    c.from_json(j);
    return std::make_shared<jimple_new>(c);
  }

  if (expr_type == "array_index")
  {
    jimple_deref c;
    c.from_json(j);
    return std::make_shared<jimple_deref>(c);
  }

  if (expr_type == "nondet")
  {
    jimple_nondet c;
    return std::make_shared<jimple_nondet>(c);
  }

  if (expr_type == "static_member")
  {
    jimple_static_member c;
    c.from_json(j.at("signature"));
    return std::make_shared<jimple_static_member>(c);
  }

  if (expr_type == "local_member")
  {
    jimple_virtual_member c;
    c.from_json(j);
    return std::make_shared<jimple_virtual_member>(c);
  }

  log_error("Unexpected expr type: {}", expr_type);
  abort();
}

void jimple_binop::from_json(const json &j)
{
  j.at("operator").get_to(binop);
  // TODO, make hashmap for each operator
  if (binop == "==")
    binop = "=";
  lhs = get_expression(j.at("lhs"));
  rhs = get_expression(j.at("rhs"));
}

// The operator reaches to_exprt as a legacy irep id -- gen_binary builds
// exprt(binop, ...) -- so the usable set is whatever migrate_expr maps, not
// whatever the Jimple producer emits. The corpus uses six: ==, +, notequal, -,
// >= and >, with from_json rewriting == to = beforehand. Anything else falls
// through to the base default and takes exactly the path it takes today, so an
// operator this switch does not know cannot silently build the wrong node.
expr2tc jimple_binop::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  expr2tc l = lhs->to_expr2t(ctx, class_name, function_name);
  expr2tc r = rhs->to_expr2t(ctx, class_name, function_name);

  // gen_binary gives the node the lhs type; the relational kinds force bool
  // themselves, which is what migrate_expr produces for them too.
  const type2tc &t = l->type;

  if (binop == "+")
    return add2tc(t, l, r);
  if (binop == "-")
    return sub2tc(t, l, r);
  if (binop == "=")
    return equality2tc(l, r);
  if (binop == "notequal")
    return notequal2tc(l, r);
  if (binop == ">")
    return greaterthan2tc(l, r);
  if (binop == ">=")
    return greaterthanequal2tc(l, r);

  return jimple_expr::to_expr2t(ctx, class_name, function_name);
}

void jimple_cast::from_json(const json &j)
{
  jimple_type type;
  j.at("to").get_to(type);
  to = std::make_shared<jimple_type>(type);
  from = get_expression(j.at("from"));
}

expr2tc jimple_cast::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  expr2tc from_expr = from->to_expr2t(ctx, class_name, function_name);
  namespacet ns(ctx);
  c_implicit_typecast(from_expr, to->to_type2t(ctx), ns);
  return from_expr;
}

void jimple_lengthof::from_json(const json &j)
{
  from = get_expression(j.at("expression"));
}

expr2tc jimple_lengthof::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  expr2tc operand = from->to_expr2t(ctx, class_name, function_name);

  symbolt lengthof = get_lengthof_function();
  symbolt &added_symbol = *ctx.move_symbol_to_context(lengthof);

  return side_effect_function_call2tc(
    migrate_type(
      static_cast<const typet &>(added_symbol.get_type().return_type())),
    symbol_expr2tc(added_symbol),
    {operand});
}

void jimple_newarray::from_json(const json &j)
{
  size = get_expression(j.at("size"));
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

void jimple_new::from_json(const json &j)
{
  size = std::make_shared<jimple_constant>("1");
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

void jimple_expr_invoke::from_json(const json &j)
{
  lhs = nil_exprt();
  j.at("base_class").get_to(base_class);
  j.at("method").get_to(method);
  for (auto x : j.at("parameters"))
  {
    parameters.push_back(std::move(jimple_expr::get_expression(x)));
  }
  method += "_" + get_hash_name();

  // TODO: Move intrinsics to backend
  if (base_class == "java.lang.Integer" && method == "valueOf_1")
  {
    log_debug("jimple", "Got an intrinsic call to valueOf int");
    is_intrinsic_method = true;
  }
}

exprt jimple_expr_invoke::to_exprt(
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

  // TODO: Move intrinsics to backend
  if (base_class == "java.lang.Integer" && method == "valueOf_1")
    // This would be called with valueOf(2), valueOf(42), etc...
    return parameters[0]->to_exprt(ctx, class_name, function_name);

  if (is_nondet_call())
  {
    jimple_nondet nondet(method);
    return nondet.to_exprt(ctx, class_name, function_name);
  }

  code_blockt block;
  code_function_callt call;

  std::ostringstream oss;
  oss << base_class << ":" << method;

  auto symbol = ctx.find_symbol(oss.str());
  if (!symbol)
  {
    log_error("Could not find symbol {}", oss.str());
    abort();
  }
  call.function() = symbol_expr(*symbol);
  if (!lhs.is_nil())
    call.lhs() = lhs;

  for (long unsigned int i = 0; i < parameters.size(); i++)
  {
    // Just adding the arguments should be enough to set the parameters
    auto parameter_expr =
      parameters[i]->to_exprt(ctx, class_name, function_name);
    call.arguments().push_back(parameter_expr);
    // Hack, manually adding parameters, this should be done at symex
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

expr2tc jimple_expr_invoke::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: Move intrinsics to backend
  if (base_class == "java.lang.Runtime")
    return code_skip2tc(get_empty_type());

  // TODO: Move intrinsics to backend
  // valueOf(n) is the identity on its argument.
  if (base_class == "java.lang.Integer" && method == "valueOf_1")
    return parameters[0]->to_expr2t(ctx, class_name, function_name);

  if (is_nondet_call())
    return jimple_nondet(method).to_expr2t(ctx, class_name, function_name);

  const std::string callee = base_class + ":" + method;
  const symbolt *symbol = ctx.find_symbol(callee);
  if (!symbol)
  {
    log_error("Could not find symbol {}", callee);
    abort();
  }

  // The legacy arm returns a block of the parameter assignments followed by the
  // call, and the assignments are its own note's "hack, manually adding
  // parameters, this should be done at symex".
  std::vector<expr2tc> ops;
  std::vector<expr2tc> args;
  args.reserve(parameters.size());
  for (std::size_t i = 0; i < parameters.size(); i++)
  {
    expr2tc arg = parameters[i]->to_expr2t(ctx, class_name, function_name);
    args.push_back(arg);

    const std::string param =
      get_symbol_name(base_class, method, "@parameter" + std::to_string(i));
    ops.push_back(code_assign2tc(symbol_expr2tc(*ctx.find_symbol(param)), arg));
  }

  ops.push_back(code_function_call2tc(lhs2, symbol_expr2tc(*symbol), args));
  const locationt &nil = static_cast<const locationt &>(get_nil_irep());
  return code_block2tc(ops, nil, nil);
}

// Restored: PR #7844 measured this arm unreached over the 27 jimple tests and
// deleted it, but jimple_assignment's virtual-invoke branch still delegates to
// the migrating default, which reaches it. Nothing in the corpus builds that
// shape, so the deletion was invisible and the branch silently produced a skip
// (docs/roadmap/scope-jimple-irep2.md §43).
exprt jimple_virtual_invoke::to_exprt(
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

  // TODO: Move intrinsics to backend
  if (base_class == "java.lang.Class")
  {
    code_skipt skip;
    return skip;
  }

  if (is_nondet_call())
  {
    jimple_nondet nondet(method);
    return nondet.to_exprt(ctx, class_name, function_name);
  }

  code_blockt block;
  code_function_callt call;

  std::ostringstream oss;
  oss << base_class << ":" << method;

  auto symbol = ctx.find_symbol(oss.str());
  call.function() = symbol_expr(*symbol);
  if (!lhs.is_nil())
  {
    call.lhs() = lhs;
  }

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

  for (long unsigned int i = 0; i < parameters.size(); i++)
  {
    // Just adding the arguments should be enough to set the parameters
    auto parameter_expr =
      parameters[i]->to_exprt(ctx, class_name, function_name);
    call.arguments().push_back(parameter_expr);
    // Hack, manually adding parameters, this should be done at symex
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

void jimple_virtual_invoke::from_json(const json &j)
{
  lhs = nil_exprt();
  j.at("base_class").get_to(base_class);
  j.at("method").get_to(method);
  j.at("name").get_to(variable);
  for (auto x : j.at("parameters"))
  {
    parameters.push_back(std::move(jimple_expr::get_expression(x)));
  }
  method += "_" + get_hash_name();
}

expr2tc jimple_virtual_invoke::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // The only arm reachable here: jimple_assignment sends an invoke right-hand
  // side to the migrating default unless it is nondet. The three skip arms and
  // the main path all produce statements, so they belong there in any case.
  if (is_nondet_call())
    return jimple_nondet(method).to_expr2t(ctx, class_name, function_name);

  return jimple_expr::to_expr2t(ctx, class_name, function_name);
}

expr2tc jimple_newarray::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  typet base_type = type->to_typet(ctx);

  // to_exprt's temp symbol only ever becomes the lhs of a call it then
  // discards, but it is still entered into the context; keep that side effect.
  symbolt tmp_symbol = get_temp_symbol(
    pointer_type2tc(migrate_type(base_type)), class_name, function_name);
  ctx.move_symbol_to_context(tmp_symbol);

  const type2tc uint2 = migrate_type(uint_type());

  expr2tc alloc_size = size->to_expr2t(ctx, class_name, function_name);
  if (is_nil_expr(alloc_size))
    alloc_size = constant_int2tc(uint2, BigInt(1));

  symbolt alloca = get_allocation_function();
  symbolt &alloca_symbol = *ctx.move_symbol_to_context(alloca);

  int type_width = 64;
  if (!(base_type.is_pointer() && base_type.subtype().is_pointer()))
    type_width = std::stoi(
      (base_type.is_pointer() ? base_type.subtype().width() : base_type.width())
        .as_string());

  expr2tc bytes =
    mul2tc(uint2, alloc_size, constant_int2tc(uint2, BigInt(type_width)));

  return side_effect_function_call2tc(
    migrate_type(
      static_cast<const typet &>(alloca_symbol.get_type().return_type())),
    symbol_expr2tc(alloca_symbol),
    {bytes});
}

void jimple_deref::from_json(const json &j)
{
  base = get_expression(j.at("base"));
  index = get_expression(j.at("index"));
}

expr2tc jimple_deref::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  expr2tc arr = base->to_expr2t(ctx, class_name, function_name);
  expr2tc offset = index->to_expr2t(ctx, class_name, function_name);

  // to_exprt assembles an index_exprt and then rewrites it in place into a
  // dereference of pointer arithmetic; this is that result, built directly.
  const type2tc &element = to_pointer_type(arr->type).subtype;
  return dereference2tc(element, add2tc(arr->type, arr, offset));
}

// gen_nondet builds exactly the sideeffect2t that migrate_expr makes of the
// legacy sideeffect("nondet") above: nil operand, size and alloc type, kind
// nondet.
expr2tc jimple_nondet::to_expr2t(
  contextt &,
  const std::string &,
  const std::string &) const
{
  return gen_nondet(migrate_type(int_type()));
}

void jimple_static_member::from_json(const json &j)
{
  j.at("base_class").get_to(from);
  j.at("member").get_to(field);
  jimple_type t;
  j.at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

expr2tc jimple_virtual_member::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // to_exprt also builds a gen_zero and looks the class tag up into a local,
  // reading neither afterwards; both are dropped here.
  expr2tc base = symbol_expr2tc(
    *ctx.find_symbol(get_symbol_name(class_name, function_name, variable)));

  if (is_pointer_type(base->type))
    base = dereference2tc(to_pointer_type(base->type).subtype, base);

  return member2tc(type->to_type2t(ctx), base, "tag-" + field);
}

expr2tc jimple_static_member::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // make_true/make_false replace the expression outright, so to_exprt's
  // gen_zero(to_typet(...)) is discarded on both of these arms.
  if (from == "kotlin._Assertions" && field == "ENABLED")
    return gen_true_expr();

  if (from == "Main" && field == "$assertionsDisabled")
    return gen_false_expr();

  // The member access itself is still marked "Needs OOP members"; leave it on
  // the migrating default.
  return jimple_expr::to_expr2t(ctx, class_name, function_name);
}

void jimple_virtual_member::from_json(const json &j)
{
  j.at("variable").get_to(variable);
  j.at("signature").at("base_class").get_to(from);
  j.at("signature").at("member").get_to(field);
  jimple_type t;
  j.at("signature").at("type").get_to(t);
  type = std::make_shared<jimple_type>(t);
}

