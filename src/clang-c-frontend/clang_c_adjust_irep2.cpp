#include <clang-c-frontend/clang_c_adjust_irep2.h>
#include <clang-c-frontend/builtin_names.h>
#include <util/irep/migrate.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
#include <util/symtab/namespace.h>
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

/// The comparisons clang_c_adjust routes through adjust_expr_rel. IREP2 already
/// types these bool, so only the operand half of that arm ports.
static bool is_relational(const expr2tc &expr)
{
  return is_equality2t(expr) || is_notequal2t(expr) || is_lessthan2t(expr) ||
         is_lessthanequal2t(expr) || is_greaterthan2t(expr) ||
         is_greaterthanequal2t(expr);
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
  {
    adjust_call_callee(expr);
    adjust_call_arguments(expr);
  }

  if (is_if2t(expr))
    adjust_if_expr(expr);

  if (is_binary_arith(expr))
    adjust_complex_arith(expr);

  if (is_relational(expr))
    adjust_relational(expr);

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

/// The lowerings `do_special_functions` selects by base name rather than by a
/// reserved `__builtin_` prefix. Returns true when `expr` was rewritten.
///
/// `sqrt`'s legacy arm additionally skips a `py:`-prefixed callee; this pass is
/// constructed only from `clang_c_languaget::typecheck`, so no Python symbol
/// can reach it and the guard has nothing to test.
bool clang_c_adjust_irep2::adjust_float_builtin(
  expr2tc &expr,
  const irep_idt &name,
  const std::vector<expr2tc> &args)
{
  const bool is_inf = compare_unscore_builtin(name, "inf") ||
                      compare_unscore_builtin(name, "huge_val");
  const bool is_nan = name != "nan" && compare_unscore_builtin(name, "nan");

  if (is_inf || is_nan)
  {
    // The fixed-point spelling is a bit pattern built off bv_width rather than
    // an ieee_floatt, and has no constant_floatbv2t to land on. Decline instead
    // of guessing: the call stays where this mode already had it.
    if (config.ansi_c.use_fixed_for_float || !is_floatbv_type(expr->type))
      return false;

    const ieee_float_spect spec(to_floatbv_type(expr->type));
    expr = constant_floatbv2tc(
      is_inf ? ieee_floatt::plus_infinity(spec) : ieee_floatt::NaN(spec));
    return true;
  }

  if (args.size() != 1)
    return false;

  const expr2tc &arg = args[0];

  if (compare_unscore_builtin(name, "isnan"))
    expr = isnan2tc(arg);
  else if (compare_unscore_builtin(name, "isinf"))
    expr = isinf2tc(arg);
  else if (compare_unscore_builtin(name, "isnormal"))
    expr = isnormal2tc(arg);
  else if (compare_unscore_builtin(name, "signbit"))
    expr = signbit2tc(arg);
  else if (
    compare_float_suffix(name, "finite") ||
    compare_unscore_builtin(name, "isfinite") ||
    compare_unscore_builtin(name, "finite"))
    expr = isfinite2tc(arg);
  else if (compare_float_suffix(name, "sqrt"))
    // The legacy node carries no rounding_mode attribute, so migrate_expr
    // synthesises this symbol for it -- the same one adjust_complex_arith
    // names.
    expr = ieee_sqrt2tc(
      expr->type, arg, symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode"));
  else if (is_abs_builtin_name(name))
  {
    // `abs` lowers to `(x >= 0) ? x : -x`, ill-typed for anything else.
    if (!is_number_type(arg->type))
      return false;
    expr = abs2tc(expr->type, arg);
  }
  else
    return false;

  return true;
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

  // A name-matched spelling the program defines itself keeps its call (#6904).
  if (builtin_shadows_user_definition(context, s->name, s->id))
    return;

  // Before the arity check: inf/huge_val/nan take no argument at all.
  if (adjust_float_builtin(expr, s->name, args))
    return;

  // The ordered-comparison builtins, which differ from the plain operators only
  // in being defined when an operand is NaN.
  if (args.size() == 2)
  {
    const expr2tc &l = args[0], &r = args[1];
    if (name == "__builtin_isgreater")
      expr = greaterthan2tc(l, r);
    else if (name == "__builtin_isgreaterequal")
      expr = greaterthanequal2tc(l, r);
    else if (name == "__builtin_isless")
      expr = lessthan2tc(l, r);
    else if (name == "__builtin_islessequal")
      expr = lessthanequal2tc(l, r);
    else if (name == "__builtin_islessgreater")
      expr = or2tc(lessthan2tc(l, r), greaterthan2tc(l, r));
    else if (name == "__builtin_isunordered")
      expr = or2tc(isnan2tc(l), isnan2tc(r));
    return;
  }

  if (args.size() != 1)
    return;

  const expr2tc &arg = args[0];

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

void clang_c_adjust_irep2::adjust_relational(expr2tc &expr)
{
  expr2tc op0 = *expr->get_sub_expr(0);
  expr2tc op1 = *expr->get_sub_expr(1);
  if (is_nil_expr(op0) || is_nil_expr(op1))
    return;

  const expr2tc before0 = op0, before1 = op1;
  c_implicit_typecast_arithmetic(op0, op1, ns);
  if (op0 == before0 && op1 == before1)
    return;

  // In-place operand surgery: never round-trip a resolved subtree through
  // migrate_expr_back (docs/roadmap/frontends-to-irep2.md §38.3).
  unsigned i = 0;
  expr->Foreach_operand([&i, &op0, &op1](expr2tc &o) { o = i++ ? op1 : op0; });
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

void clang_c_adjust_irep2::adjust_call_arguments(expr2tc &expr)
{
  expr2tc callee;
  std::vector<expr2tc> *args;
  if (is_code_function_call2t(expr))
  {
    code_function_call2t &call = to_code_function_call2t(expr);
    callee = call.function;
    args = &call.operands;
  }
  else
  {
    sideeffect2t &se = to_sideeffect2t(expr);
    if (se.kind != sideeffect_allockind::function_call)
      return;
    callee = se.operand;
    args = &se.arguments;
  }

  if (is_nil_expr(callee))
    return;

  type2tc ct = callee->type;
  if (is_pointer_type(ct))
    ct = to_pointer_type(ct).subtype;
  if (!is_code_type(ct))
    return;

  const std::vector<type2tc> &params = to_code_type(ct).arguments;

  for (std::size_t i = 0; i < args->size(); i++)
  {
    expr2tc &arg = (*args)[i];
    if (is_nil_expr(arg))
      continue;

    if (i < params.size())
      c_implicit_typecast(arg, params[i], ns);
    else if (is_array_type(ns.follow(arg->type)))
      // A variadic argument has no parameter type to convert against; only the
      // array decay is owed.
      c_implicit_typecast(arg, pointer_type2tc(get_empty_type()), ns);
  }
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

  symbolt sym;
  sym.id = id;
  sym.name = id;
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
