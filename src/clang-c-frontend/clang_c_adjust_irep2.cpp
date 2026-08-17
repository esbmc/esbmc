#include <clang-c-frontend/clang_c_adjust_irep2.h>
#include <util/irep/migrate.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/namespace.h>
#include <util/symtab/pretty.h>
#include <utility>

bool clang_c_adjust_irep2::adjust()
{
  // migrate_expr resolves a symbol's type through this thread-local namespace.
  // typecheck builds into its own context and c_link merges into the global one
  // afterwards, so the namespace language_ui installed does not contain these
  // symbols yet: every lookup would miss and fall through to
  // sym_name_to_symbol's renaming parser, which warns once per symbol. Point it
  // at the context being built, as dereferencet does for the same reason.
  namespacet ns(context);
  const namespacet *old_ns = std::exchange(migrate_namespace_lookup, &ns);

  // Hash-table iterators are not stable across mutation, so snapshot the
  // symbol pointers first (mirrors clang_c_adjust::adjust()).
  std::vector<symbolt *> symbol_list;
  context.Foreach_operand_in_order(
    [&symbol_list](symbolt &s) { symbol_list.push_back(&s); });

  for (symbolt *s : symbol_list)
  {
    if (!s->is_type && s->get_value().is_not_nil())
    {
      const expr2tc before = s->get_value2();
      expr2tc value = before;
      adjust_expr(value);
      // Only write back a value this pass actually changed, so symbols it does
      // not touch never make the round trip (python_adjust takes the same care,
      // for the bitfield and alignment losses migrate_type cannot carry).
      if (value != before)
        s->set_value(value);
    }
  }

  migrate_namespace_lookup = old_ns;
  return false;
}

/// The operators C admits over a complex operand: `mod` and the bitwise ones
/// are not among them, and `clang_c_adjust` aborts rather than lowering those.
static bool is_binary_arith(const expr2tc &expr)
{
  return is_add2t(expr) || is_sub2t(expr) || is_mul2t(expr) || is_div2t(expr);
}

/// `-z` and GNU `~z` (conjugation) are the only unary operators clang leaves
/// carrying a complex type.
static bool is_complex_unary(const expr2tc &expr)
{
  return (is_neg2t(expr) || is_bitnot2t(expr)) && is_complex_type(expr->type);
}

void clang_c_adjust_irep2::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });

  if (is_index2t(expr))
    adjust_index(expr);
  else if (is_member2t(expr))
    adjust_member(expr);
  else if (
    sole_adjuster && (is_code_function_call2t(expr) || is_sideeffect2t(expr)))
    declare_implicit_callee(expr);

  if (sole_adjuster)
    adjust_sole_arms(expr);
}

/// The arms that only run when this pass is the sole adjuster, gathered behind
/// one test so adjust_expr does not repeat it per arm.
void clang_c_adjust_irep2::adjust_sole_arms(expr2tc &expr)
{
  if (is_and2t(expr) || is_or2t(expr) || is_not2t(expr))
    adjust_boolean_operands(expr);

  if (is_code_function_call2t(expr) || is_sideeffect2t(expr))
    adjust_call_callee(expr);

  if (is_if2t(expr))
    adjust_if_expr(expr);

  if (is_binary_arith(expr))
    adjust_complex_arith(expr);

  if (is_complex_unary(expr))
    adjust_complex_unary(expr);

  if (is_sideeffect2t(expr))
    adjust_special_functions(expr);
}

/// One of a family of spellings differing only by the argument's width:
/// `__builtin_popcount`, `...l`, `...ll`. The lowering is the same node for all
/// three, so the suffix carries no information here.
static bool is_width_suffixed(const std::string &name, const std::string &stem)
{
  if (name.compare(0, stem.size(), stem) != 0)
    return false;
  const std::string suffix = name.substr(stem.size());
  return suffix.empty() || suffix == "l" || suffix == "ll";
}

/// The one-argument builtins that lower to a single IREP2 node.
static void
fold_unary_builtin(const std::string &name, const expr2tc &arg, expr2tc &expr)
{
  if (
    is_width_suffixed(name, "__builtin_popcount") || name == "__popcnt" ||
    name == "__popcnt16" || name == "__popcnt64")
    expr = popcount2tc(arg);
  else if (is_width_suffixed(name, "__builtin_parity"))
    // parity(x) = popcount(x) & 1.
    expr = bitand2tc(
      get_int32_type(), popcount2tc(arg), constant_int2tc(get_int32_type(), 1));
  else if (
    name == "__builtin_bswap16" || name == "__builtin_bswap32" ||
    name == "__builtin_bswap64")
    expr = bswap2tc(expr->type, arg);
}

void clang_c_adjust_irep2::adjust_special_functions(expr2tc &expr)
{
  const sideeffect2t &se = to_sideeffect2t(expr);
  if (
    se.kind != sideeffect_allockind::function_call || is_nil_expr(se.operand) ||
    !is_symbol2t(se.operand))
    return;

  // symbol2t carries the linkage identifier (`c:@F@__builtin_expect`), not the
  // base name do_special_functions matches on; the symbol table holds both, and
  // builtin_functions.cpp reads the base name the same way.
  const symbolt *s = context.find_symbol(to_symbol2t(se.operand).thename);
  if (s == nullptr)
    return;

  const std::string name = id2string(s->name);
  const std::vector<expr2tc> &args = se.arguments;

  // A branch-prediction hint evaluates to its first argument. Left as a call it
  // is bodyless, so its result is nondet -- and Darwin's assert.h expands
  // assert(e) through it, which makes every such assertion nondet rather than
  // merely differently-shaped (§90).
  if (name == "__builtin_expect" && args.size() == 2)
  {
    expr = args[0];
    return;
  }

  if (args.size() == 1)
    fold_unary_builtin(name, args[0], expr);
}

void clang_c_adjust_irep2::adjust_if_expr(expr2tc &expr)
{
  const if2t &i = to_if2t(expr);
  expr2tc cond = i.cond, tv = i.true_value, fv = i.false_value;

  if (!is_nil_expr(cond) && !is_bool_type(cond->type))
    c_implicit_typecast(cond, get_bool_type(), ns);

  if (!is_nil_expr(tv) && tv->type != expr->type)
    c_implicit_typecast(tv, expr->type, ns);
  if (!is_nil_expr(fv) && fv->type != expr->type)
    c_implicit_typecast(fv, expr->type, ns);

  if (cond != i.cond || tv != i.true_value || fv != i.false_value)
    expr = if2tc(expr->type, cond, tv, fv, i.location);
}

void clang_c_adjust_irep2::adjust_call_callee(expr2tc &expr)
{
  expr2tc callee;
  if (is_code_function_call2t(expr))
    callee = to_code_function_call2t(expr).function;
  else
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    if (se.kind != sideeffect_allockind::function_call)
      return;
    callee = se.operand;
  }

  if (is_nil_expr(callee) || !is_pointer_type(callee->type))
    return;

  const expr2tc deref =
    dereference2tc(to_pointer_type(callee->type).subtype, callee);

  if (is_code_function_call2t(expr))
    to_code_function_call2t(expr).function = deref;
  else
    to_sideeffect2t(expr).operand = deref;
}

void clang_c_adjust_irep2::adjust_boolean_operands(expr2tc &expr)
{
  expr->Foreach_operand([this](expr2tc &op) {
    if (!is_nil_expr(op) && !is_bool_type(op->type))
      c_implicit_typecast(op, get_bool_type(), ns);
  });
}

static bool contains_sideeffect(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return false;
  if (is_sideeffect2t(expr))
    return true;

  bool found = false;
  expr->foreach_operand(
    [&found](const expr2tc &op) { found = found || contains_sideeffect(op); });
  return found;
}

void clang_c_adjust_irep2::adjust_complex_arith(expr2tc &expr)
{
  expr2tc op0 = *expr->get_sub_expr(0);
  expr2tc op1 = *expr->get_sub_expr(1);

  if (
    is_nil_expr(op0) || is_nil_expr(op1) ||
    (!is_complex_type(op0->type) && !is_complex_type(op1->type)))
    return;

  // Each operand is read twice below, once per component, so lowering one that
  // performs a side effect would evaluate it twice -- a wrong verdict, where
  // declining only leaves the node where this mode already had it. The binding
  // clang_c_adjust does first (a context temporary plus a statement
  // expression) is unported; §88.2 records why porting it is separate work.
  if (contains_sideeffect(op0) || contains_sideeffect(op1))
    return;

  const type2tc ct = is_complex_type(op0->type) ? op0->type : op1->type;
  const type2tc et = to_complex_type(ct).subtype;

  auto promote = [&ct, &et](expr2tc &e) {
    if (!is_complex_type(e->type))
      e = constant_struct2tc(ct, std::vector<expr2tc>{e, gen_zero(et)});
  };
  promote(op0);
  promote(op1);

  // migrate_expr synthesises the same rounding-mode symbol for a legacy
  // ieee_* node that carries none, which is what clang_c_adjust emits here.
  const expr2tc rm = symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");
  const bool fp = is_floatbv_type(et);

  auto mk = [&et, &rm, fp](char op, const expr2tc &l, const expr2tc &r) {
    switch (op)
    {
    case '+':
      return fp ? expr2tc(ieee_add2tc(et, l, r, rm))
                : expr2tc(add2tc(et, l, r));
    case '-':
      return fp ? expr2tc(ieee_sub2tc(et, l, r, rm))
                : expr2tc(sub2tc(et, l, r));
    case '*':
      return fp ? expr2tc(ieee_mul2tc(et, l, r, rm))
                : expr2tc(mul2tc(et, l, r));
    default:
      return fp ? expr2tc(ieee_div2tc(et, l, r, rm))
                : expr2tc(div2tc(et, l, r));
    }
  };

  const expr2tc ar = member2tc(et, op0, "real");
  const expr2tc ai = member2tc(et, op0, "imag");
  const expr2tc br = member2tc(et, op1, "real");
  const expr2tc bi = member2tc(et, op1, "imag");

  expr2tc re, im;
  switch (expr->expr_id)
  {
  case expr2t::add_id:
    re = mk('+', ar, br);
    im = mk('+', ai, bi);
    break;

  case expr2t::sub_id:
    re = mk('-', ar, br);
    im = mk('-', ai, bi);
    break;

  case expr2t::mul_id:
    re = mk('-', mk('*', ar, br), mk('*', ai, bi));
    im = mk('+', mk('*', ar, bi), mk('*', ai, br));
    break;

  default:
  {
    assert(is_div2t(expr));
    const expr2tc denom = mk('+', mk('*', br, br), mk('*', bi, bi));
    re = mk('/', mk('+', mk('*', ar, br), mk('*', ai, bi)), denom);
    im = mk('/', mk('-', mk('*', ai, br), mk('*', ar, bi)), denom);
    break;
  }
  }

  expr = constant_struct2tc(ct, std::vector<expr2tc>{re, im});
}

void clang_c_adjust_irep2::adjust_complex_unary(expr2tc &expr)
{
  const expr2tc op = *expr->get_sub_expr(0);

  // Same double-evaluation decline as adjust_complex_arith (§88.2, §90.2).
  if (contains_sideeffect(op))
    return;

  const type2tc ct = expr->type;
  const type2tc et = to_complex_type(ct).subtype;

  // No ieee_ form is needed here, unlike the binary operators: negation is a
  // sign-bit flip, exact and independent of the rounding mode.
  expr2tc re = member2tc(et, op, "real");
  if (is_neg2t(expr))
    re = neg2tc(et, re);
  const expr2tc im = neg2tc(et, member2tc(et, op, "imag"));

  expr = constant_struct2tc(ct, std::vector<expr2tc>{re, im});
}

void clang_c_adjust_irep2::declare_implicit_callee(const expr2tc &expr)
{
  // A bare `f(x);` statement is a sideeffect2t of kind function_call, not a
  // code_function_call2t; both spellings reach here.
  expr2tc callee;
  locationt loc;
  if (is_code_function_call2t(expr))
  {
    const code_function_call2t &call = to_code_function_call2t(expr);
    callee = call.function;
    loc = call.location;
  }
  else
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    if (se.kind != sideeffect_allockind::function_call)
      return;
    callee = se.operand;
  }

  if (is_nil_expr(callee) || !is_symbol2t(callee))
    return;

  const irep_idt id = to_symbol2t(callee).thename;
  if (context.find_symbol(id) != nullptr)
    return;

  // symbol2t carries only the linkage identifier, so the base name
  // clang_c_adjust copies off the symbol expression (`f_op.name()`) has to be
  // recovered from it. do_function_call_symbol matches
  // `assert`/`__ESBMC_assume` and the rest on the base name, so leaving the
  // identifier here leaves an assert as a plain FUNCTION_CALL (§90.2).
  symbolt sym;
  sym.id = id;
  sym.name = get_pretty_name(id2string(id));
  sym.location = loc;
  sym.set_type(migrate_type_back(callee->type));
  sym.mode = "C";
  context.add(sym);
}

void adjust_comma_at_dispatch(exprt &expr, const namespacet &ns)
{
  const namespacet *old_ns = std::exchange(migrate_namespace_lookup, &ns);

  expr2tc e;
  migrate_expr(expr, e);
  const code_comma2t &c = to_code_comma2t(e);
  expr = migrate_expr_back(code_comma2tc(c.side_2->type, c.side_1, c.side_2));

  migrate_namespace_lookup = old_ns;
}

void clang_c_adjust_irep2::adjust_member(expr2tc &expr)
{
  const member2t &m = to_member2t(expr);
  expr2tc base = m.source_value;

  if (is_pointer_type(base->type))
    base = dereference2tc(to_pointer_type(base->type).subtype, base);
  else if (is_array_type(base->type))
    base = index2tc(
      to_array_type(base->type).subtype,
      base,
      gen_zero(migrate_type(index_type())));
  else
    return;

  expr = member2tc(expr->type, base, m.member);
}

void clang_c_adjust_irep2::adjust_index(expr2tc &expr)
{
  const index2t &idx = to_index2t(expr);
  expr2tc array_expr = idx.source_value;
  expr2tc index_expr = idx.index;

  // The operands may be the other way round: `i[a]` is legal C.
  if (const type2tc a = ns.follow(array_expr->type),
      i = ns.follow(index_expr->type);
      !is_array_type(a) && !is_pointer_type(a) &&
      (is_array_type(i) || is_pointer_type(i)))
    std::swap(array_expr, index_expr);

  // migrate_type(index_type()), not index_type2(): despite the name they are
  // different types -- index_type() is signed_size_type(), index_type2() is
  // get_int_type(config.ansi_c.address_width) -- so the latter leaves a
  // non-constant subscript uncast where the legacy arm casts it.
  c_implicit_typecast(index_expr, migrate_type(index_type()), ns);

  const type2tc final_array_type = ns.follow(array_expr->type);

  // p[i] is syntactic sugar for *(p+i).
  if (is_pointer_type(final_array_type))
  {
    expr = dereference2tc(
      expr->type, add2tc(array_expr->type, array_expr, index_expr));
    return;
  }

  if (!is_array_type(final_array_type) && !is_vector_type(final_array_type))
  {
    // The base is neither array, vector nor pointer -- typically a struct that
    // appears in array context because two TUs declared the same external
    // symbol with conflicting types. Rewrite `base[i]` as `*((T*)&base + i)`.
    const type2tc elem_type = expr->type;
    const type2tc ptr_type = pointer_type2tc(elem_type);

    expr2tc addr_of =
      address_of2tc(pointer_type2tc(array_expr->type), array_expr);
    c_implicit_typecast(addr_of, ptr_type, ns);

    expr = dereference2tc(elem_type, add2tc(ptr_type, addr_of, index_expr));
    return;
  }

  expr = index2tc(expr->type, array_expr, index_expr);
}
