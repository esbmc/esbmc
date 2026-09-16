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

// gen_binary gives the node the lhs type; these kinds keep it.
static expr2tc jimple_typed_binop(
  const std::string &op,
  const type2tc &t,
  const expr2tc &l,
  const expr2tc &r)
{
  if (op == "+")
    return add2tc(t, l, r);
  if (op == "-")
    return sub2tc(t, l, r);
  if (op == "*")
    return mul2tc(t, l, r);
  if (op == "/")
    return div2tc(t, l, r);
  if (op == "mod")
    return modulus2tc(t, l, r);
  if (op == "bitand")
    return bitand2tc(t, l, r);
  if (op == "bitor")
    return bitor2tc(t, l, r);
  if (op == "bitxor")
    return bitxor2tc(t, l, r);
  if (op == "shl")
    return shl2tc(t, l, r);
  if (op == "ashr")
    return ashr2tc(t, l, r);
  // Mirrors migrate_expr's arm rather than a test: jimple builds no unsigned
  // type, so a logical and an arithmetic shift right of a signed operand print
  // the same and agree on every verdict -- swapping the two changes nothing
  // observable (§38.2).
  if (op == "lshr")
    return lshr2tc(t, l, r);
  return expr2tc();
}

// The relational kinds force bool themselves, which is what migrate_expr
// produces for them too.
static expr2tc jimple_relational_binop(
  const std::string &op,
  const expr2tc &l,
  const expr2tc &r)
{
  if (op == "=")
    return equality2tc(l, r);
  if (op == "notequal")
    return notequal2tc(l, r);
  if (op == "<")
    return lessthan2tc(l, r);
  if (op == "<=")
    return lessthanequal2tc(l, r);
  if (op == ">")
    return greaterthan2tc(l, r);
  if (op == ">=")
    return greaterthanequal2tc(l, r);
  return expr2tc();
}

expr2tc jimple_binop::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  expr2tc l = lhs->to_expr2t(ctx, class_name, function_name);
  expr2tc r = rhs->to_expr2t(ctx, class_name, function_name);

  expr2tc e = jimple_typed_binop(binop, l->type, l, r);
  if (is_nil_expr(e))
    e = jimple_relational_binop(binop, l, r);
  if (!is_nil_expr(e))
    return e;

  // Both representations require these to be bool throughout: migrate_expr
  // asserts the legacy node's type is bool, and goto_check asserts the node and
  // *each operand* are (goto_check.cpp, and_id/or_id). The legacy arm handed
  // gen_binary the lhs type, so a jimple `and` over two ints aborted an
  // assert-enabled build in either representation -- which no NDEBUG build and
  // no test in the corpus could show (§38.5).
  if (binop == "and" || binop == "or")
  {
    namespacet ns(ctx);
    c_implicit_typecast(l, get_bool_type(), ns);
    c_implicit_typecast(r, get_bool_type(), ns);
    return binop == "and" ? expr2tc(and2tc(l, r)) : expr2tc(or2tc(l, r));
  }

  // Every spelling the frontend converts end to end is covered above
  // (scope-jimple-irep2.md §38.1). Rejecting the rest here rather than letting
  // migrate_expr reject them names the operator instead of the irep id, and it
  // is what leaves no caller for any expression to_exprt.
  throw "Unsupported Jimple operator: " + binop;
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
  lhs = expr2tc();
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

// The legacy arms built their statements without a location, and migrate_expr
// read that absent `#location` through the const accessor, i.e. as nil -- which
// goto_programt prints as "no location", where a default-constructed locationt
// is empty-but-not-nil and prints blank. goto_convert_functions'
// emitted_location documents the same distinction from the other side.
static locationt no_location()
{
  locationt l;
  l.make_nil();
  return l;
}

static expr2tc skip2t_without_location()
{
  return code_skip2tc(get_empty_type(), no_location());
}

expr2tc jimple_expr::lower_invoke2t(
  contextt &ctx,
  const std::string &base_class,
  const std::string &method,
  const std::string &this_variable,
  const std::vector<std::shared_ptr<jimple_expr>> &parameters,
  const expr2tc &lhs,
  const std::string &class_name,
  const std::string &function_name)
{
  const std::string callee_id = base_class + ":" + method;
  const symbolt *callee = ctx.find_symbol(callee_id);
  if (callee == nullptr)
  {
    log_error("Could not find symbol {}", callee_id);
    abort();
  }

  const locationt none = no_location();

  std::vector<expr2tc> stmts;
  std::vector<expr2tc> args;

  auto bind = [&](const std::string &bound_name, const expr2tc &value) {
    args.push_back(value);
    const symbolt &bound =
      *ctx.find_symbol(get_symbol_name(base_class, method, bound_name));
    stmts.push_back(code_assign2tc(symbol_expr2tc(bound), value, none));
  };

  if (!this_variable.empty())
    bind(
      "@this",
      jimple_symbol(this_variable).to_expr2t(ctx, class_name, function_name));

  for (std::size_t i = 0; i < parameters.size(); i++)
    bind(
      "@parameter" + std::to_string(i),
      parameters[i]->to_expr2t(ctx, class_name, function_name));

  stmts.push_back(
    code_function_call2tc(lhs, symbol_expr2tc(*callee), args, none));
  return code_block2tc(stmts, none, none);
}

expr2tc jimple_expr_invoke::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  // TODO: Move intrinsics to backend
  if (
    base_class == "kotlin.jvm.internal.Intrinsics" ||
    base_class == "java.lang.Runtime")
    return skip2t_without_location();

  // TODO: Move intrinsics to backend
  // valueOf(n) is the identity on its argument.
  if (base_class == "java.lang.Integer" && method == "valueOf_1")
    return parameters[0]->to_expr2t(ctx, class_name, function_name);

  if (is_nondet_call())
    return jimple_nondet(method).to_expr2t(ctx, class_name, function_name);

  return lower_invoke2t(
    ctx, base_class, method, "", parameters, lhs, class_name, function_name);
}

void jimple_virtual_invoke::from_json(const json &j)
{
  lhs = expr2tc();
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
  // TODO: Move intrinsics to backend
  if (
    base_class == "kotlin.jvm.internal.Intrinsics" ||
    base_class == "java.lang.Runtime" || base_class == "java.lang.Class")
    return skip2t_without_location();

  if (is_nondet_call())
    return jimple_nondet(method).to_expr2t(ctx, class_name, function_name);

  return lower_invoke2t(
    ctx,
    base_class,
    method,
    variable,
    parameters,
    lhs,
    class_name,
    function_name);
}

expr2tc jimple_newarray::to_expr2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  const type2tc base_type = type->to_type2t(ctx);

  // to_exprt's temp symbol only ever becomes the lhs of a call it then
  // discards, but it is still entered into the context; keep that side effect.
  symbolt tmp_symbol =
    get_temp_symbol(pointer_type2tc(base_type), class_name, function_name);
  ctx.move_symbol_to_context(tmp_symbol);

  const type2tc uint2 = uint_type2();

  expr2tc alloc_size = size->to_expr2t(ctx, class_name, function_name);
  if (is_nil_expr(alloc_size))
    alloc_size = constant_int2tc(uint2, BigInt(1));

  symbolt alloca = get_allocation_function();
  symbolt &alloca_symbol = *ctx.move_symbol_to_context(alloca);

  // A row of a multi-dimensional array is a pointer. Keep the literal 64 the
  // legacy arm used rather than the pointer type's own width, which would
  // change the allocation on a 32-bit target.
  const type2tc &element =
    is_pointer_type(base_type) ? to_pointer_type(base_type).subtype : base_type;
  unsigned int type_width =
    is_pointer_type(element) ? 64 : element->get_width();

  expr2tc bytes =
    mul2tc(uint2, alloc_size, constant_int2tc(uint2, BigInt(type_width)));

  return side_effect_function_call2tc(
    to_code_type(alloca_symbol.get_type2()).ret_type,
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

exprt jimple_static_member::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  auto result = gen_zero(type->to_typet(ctx));
  // HACK: For now I will set some intrinsics directly (this should go to SYMEX)
  if (from == "kotlin._Assertions" && field == "ENABLED")
  {
    result.make_true();
    return result;
  }

  if (from == "Main" && field == "$assertionsDisabled")
  {
    result.make_false();
    return result;
  }

  // TODO: Needs OOP members

  // 1. Look over the local scope
  auto symbol_name = get_symbol_name(class_name, function_name, from);
  symbolt &s = *ctx.find_symbol(symbol_name);
  member_exprt op(symbol_expr(s), "tag-" + field, s.get_type());
  exprt &base = op.struct_op();
  if (base.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
  return op;
};

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
