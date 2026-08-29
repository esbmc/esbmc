#include <clang-c-frontend/clang_c_adjust.h>
#include <clang-c-frontend/clang_c_adjust_irep2.h>
#include <clang-c-frontend/padding.h>
#include <clang-c-frontend/builtin_names.h>
#include <util/irep/migrate.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/config/config.h>
#include <util/symtab/namespace.h>
#include <util/symtab/pretty.h>
#include <util/symtab/cprover_prefix.h>
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

  // Types first, in a pass of their own: a value's initialiser is built from
  // its type, so padding a type after a value that uses it leaves the value
  // short a component. clang_c_adjust::adjust() splits the walk for the same
  // reason ("so that symbolic-type resolution always receives fixed up types").
  if (sole_adjuster)
    for (symbolt *s : symbol_list)
      if (s->is_type)
        pad_type_symbol(*s);

  for (symbolt *s : symbol_list)
  {
    if (
      sole_adjuster && s->get_type().is_code() &&
      has_prefix(s->id.as_string(), "c:@F@main"))
      declare_argc_argv(context, *s);

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

/// add_padding on a complete struct or union, which is the half of
/// clang_c_adjust::adjust_type that the corpus shows is load-bearing here. The
/// function is shared (clang-c-frontend/padding.h) and idempotent --
/// adjust_type asserts that re-padding is a no-op -- so this reuses it rather
/// than reimplementing a layout algorithm over type2tc.
void clang_c_adjust_irep2::pad_type_symbol(symbolt &symbol)
{
  typet t = symbol.get_type();
  if ((!t.is_struct() && !t.is_union()) || t.incomplete())
    return;

  add_padding(t, ns);
  symbol.set_type(std::move(t));
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

/// The operators clang_c_adjust routes through adjust_expr_binary_arithmetic.
static bool is_arith_or_bitwise(const expr2tc &expr)
{
  return is_binary_arith(expr) || is_modulus2t(expr) || is_bitand2t(expr) ||
         is_bitor2t(expr) || is_bitxor2t(expr);
}

/// The statements whose controlling expression clang_c_adjust converts to bool
/// (adjust_ifthenelse, adjust_while, adjust_for). `switch` is not among them:
/// its selector is an integer.
static bool is_statement_with_condition(const expr2tc &expr)
{
  return is_code_ifthenelse2t(expr) || is_code_while2t(expr) ||
         is_code_dowhile2t(expr) || is_code_for2t(expr);
}

/// The comparisons clang_c_adjust routes through adjust_expr_rel. IREP2 already
/// types these bool, so only the operand half of that arm ports.
static bool is_relational(const expr2tc &expr)
{
  return is_equality2t(expr) || is_notequal2t(expr) || is_lessthan2t(expr) ||
         is_lessthanequal2t(expr) || is_greaterthan2t(expr) ||
         is_greaterthanequal2t(expr);
}

/// The source location of `expr` when it is a statement that can hold a call
/// in a sub-expression, empty otherwise. sideeffect2t carries none of its own,
/// so a call in one takes the enclosing statement's -- which is what
/// clang_c_adjust reads off the side_effect_expr_function_callt it is handed
/// (§130.3). Every kind whose operands can be an expression is listed: a
/// condition is as much a call site as an assignment's right-hand side, and
/// omitting one leaves the counterexample with no file and line 0.
static locationt statement_location(const expr2tc &expr)
{
  switch (expr->expr_id)
  {
  case expr2t::code_expression_id:
    return to_code_expression2t(expr).location;
  case expr2t::code_assign_id:
    return to_code_assign2t(expr).location;
  case expr2t::code_decl_id:
    return to_code_decl2t(expr).location;
  case expr2t::code_return_id:
    return to_code_return2t(expr).location;
  case expr2t::code_function_call_id:
    return to_code_function_call2t(expr).location;
  case expr2t::code_ifthenelse_id:
    return to_code_ifthenelse2t(expr).location;
  case expr2t::code_while_id:
    return to_code_while2t(expr).location;
  case expr2t::code_dowhile_id:
    return to_code_dowhile2t(expr).location;
  case expr2t::code_for_id:
    return to_code_for2t(expr).location;
  case expr2t::code_switch_id:
    return to_code_switch2t(expr).location;
  case expr2t::code_assert_id:
    return to_code_assert2t(expr).location;
  case expr2t::code_assume_id:
    return to_code_assume2t(expr).location;
  case expr2t::code_printf_id:
    return to_code_printf2t(expr).location;
  default:
    return locationt();
  }
}

void clang_c_adjust_irep2::adjust_expr(expr2tc &expr)
{
  if (is_nil_expr(expr))
    return;

  // Before the recursion, so the located spelling wins over the unlocated one
  // the walk would otherwise reach first.
  if (sole_adjuster && is_code_expression2t(expr))
  {
    const code_expression2t &stmt = to_code_expression2t(expr);
    declare_implicit_callee(stmt.operand, stmt.location);
  }

  // A call reached below a statement takes that statement's location; a
  // sideeffect2t has none of its own.
  const locationt saved_location = enclosing_location;
  if (const locationt l = statement_location(expr); !l.get_line().empty())
    enclosing_location = l;

  expr->Foreach_operand([this](expr2tc &op) { adjust_expr(op); });

  if (is_index2t(expr))
    adjust_index(expr);
  else if (is_member2t(expr))
    adjust_member(expr);
  else if (
    sole_adjuster && (is_code_function_call2t(expr) || is_sideeffect2t(expr)))
  {
    // Before declare_implicit_callee: the polymorphic name is repointed at the
    // concrete instance here, and it is that symbol the callee check must see.
    // Before adjust_function_designators too, which wraps a code-typed callee
    // in an address_of2t that the symbol test below would then reject.
    declare_polymorphic_builtin(expr);
    declare_implicit_callee(expr);
  }

  if (sole_adjuster)
    adjust_sole_arms(expr);

  if (sole_adjuster && is_address_of2t(expr))
    adjust_address_of(expr);

  enclosing_location = saved_location;
}

/// The arms that only run when this pass is the sole adjuster, gathered behind
/// one test so adjust_expr does not repeat it per arm.
void clang_c_adjust_irep2::adjust_sole_arms(expr2tc &expr)
{
  // First: the sugar has to be in place before adjust_call_callee decides
  // whether this call is direct, since that is what it reads.
  adjust_function_designators(expr);

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
  {
    adjust_complex_arith(expr);
    adjust_vector_float_arith(expr);
  }

  /* Before the hoist: hoist_for_init rewrites a code_for2t into a block, and a
   * block is not a statement-with-condition, so the loop's guard would never
   * reach the conversion. */
  if (is_statement_with_condition(expr))
    adjust_statement_condition(expr);

  if (is_code_for2t(expr))
    hoist_for_init(expr);

  if (is_code_expression2t(expr))
    adjust_expression_statement(expr);

  adjust_sole_arms_tail(expr);
}

/// The tail of adjust_sole_arms. Split only to keep either half under
/// the complexity gate; the two run back to back and the arms below are
/// order-independent of the ones above.
void clang_c_adjust_irep2::adjust_sole_arms_tail(expr2tc &expr)
{
  // A comma expression takes its right operand's type (C11 6.5.17p2). Clang
  // hands it the *decayed* type when the right operand is an array, so leaving
  // it makes `(c, a[i])[0]` index a pointer rather than the row -- which loses
  // the named array-bounds check for the generic dereference one. Same rewrite
  // as adjust_comma_at_dispatch, which the --clang-c-irep2-adjust probe uses.
  if (is_code_comma2t(expr))
  {
    const code_comma2t &c = to_code_comma2t(expr);
    if (expr->type != c.side_2->type)
      expr = code_comma2tc(c.side_2->type, c.side_1, c.side_2);
  }
  if (is_constant_struct2t(expr))
    adjust_struct(expr);
  if (is_constant_array2t(expr))
    adjust_array_subtype(expr);
  if (is_code_decl2t(expr))
    adjust_decl_init(expr);
  if (is_dereference2t(expr))
    adjust_dereference(expr);

  if (is_complex_unary(expr))
    adjust_complex_unary(expr);
  else if (is_neg2t(expr) || is_bitnot2t(expr))
    promote_unary_bool_operand(expr);

  if (is_relational(expr))
    adjust_relational(expr);

  if (is_sideeffect2t(expr))
    adjust_special_functions(expr);

  if (is_arith_or_bitwise(expr))
    adjust_binary_arith_operands(expr);

  if (is_sideeffect_assign2t(expr))
  {
    adjust_plain_assignment(expr);
    adjust_compound_assignment(expr);
  }
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

/// The ordered-comparison builtins, which differ from the plain operators only
/// in being defined when an operand is NaN.
static void fold_comparison_builtin(
  const std::string &name,
  const expr2tc &l,
  const expr2tc &r,
  expr2tc &expr)
{
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

/// The three pointer intrinsics `do_special_functions` matches by their
/// `__ESBMC_` name rather than a `__builtin_` prefix. Each lowers to a node the
/// backend evaluates in place; left as a call the symbol is bodyless and
/// `goto_check`'s "non-intrinsic prefixed with __ESBMC" rejects the program.
/// Returns true when \p expr was rewritten.
static bool fold_pointer_intrinsic(
  const std::string &name,
  const std::vector<expr2tc> &args,
  expr2tc &expr)
{
  if (name == CPROVER_PREFIX "same_object" && args.size() == 2)
  {
    expr = same_object2tc(args[0], args[1]);
    return true;
  }

  if (args.size() != 1)
    return false;

  if (name == CPROVER_PREFIX "POINTER_OBJECT")
  {
    expr = pointer_object2tc(expr->type, args[0]);
    return true;
  }

  // pointer_offset2t admits only an address-width signedbv. The declared
  // return type is __PTRDIFF_TYPE__, which is exactly that on every supported
  // target; decline rather than assert if a target ever disagrees.
  if (
    name == CPROVER_PREFIX "POINTER_OFFSET" && is_signedbv_type(expr->type) &&
    expr->type->get_width() == config.ansi_c.address_width)
  {
    expr = pointer_offset2tc(expr->type, args[0]);
    return true;
  }

  return false;
}

/// The lowerings `do_special_functions` selects by base name rather than by a
/// reserved `__builtin_` prefix. Returns true when `expr` was rewritten.
///
/// `sqrt`'s legacy arm additionally skips a `py:`-prefixed callee; this pass is
/// constructed only from `clang_c_languaget::typecheck`, so no Python symbol
/// can reach it and the guard has nothing to test.
/// The argument-less float constants. `handled` distinguishes "not one of
/// these" from "one of these, but declined".
static bool
fold_float_constant(expr2tc &expr, const irep_idt &name, bool &handled)
{
  const bool is_inf = compare_unscore_builtin(name, "inf") ||
                      compare_unscore_builtin(name, "huge_val");
  const bool is_nan = name != "nan" && compare_unscore_builtin(name, "nan");

  handled = is_inf || is_nan;
  if (!handled)
    return false;

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

/// `sqrt`'s legacy arm additionally skips a `py:`-prefixed callee; this pass is
/// constructed only from `clang_c_languaget::typecheck`, so no Python symbol
/// can reach it and the guard has nothing to test.
bool clang_c_adjust_irep2::adjust_float_builtin(
  expr2tc &expr,
  const irep_idt &name,
  const std::vector<expr2tc> &args)
{
  bool handled = false;
  if (const bool folded = fold_float_constant(expr, name, handled); handled)
    return folded;

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
  // Exact spelling: `isinf_sign` is reserved, and compare_unscore_builtin's
  // "isinf" arm above matches a base name a program may reuse. Left as a call
  // the symbol is bodyless, so the result is nondet rather than differently
  // shaped -- the flag turns a provable comparison into an unprovable one.
  else if (name == "__builtin_isinf_sign")
    expr = if2tc(
      expr->type,
      isinf2tc(arg),
      if2tc(
        expr->type,
        typecast2tc(get_bool_type(), signbit2tc(arg)),
        constant_int2tc(expr->type, BigInt(-1)),
        constant_int2tc(expr->type, BigInt(1))),
      gen_zero(expr->type));
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

  if (fold_pointer_intrinsic(name, args, expr))
    return;

  if (args.size() == 2)
  {
    fold_comparison_builtin(name, args[0], args[1], expr);
    return;
  }

  if (args.size() == 1)
    fold_unary_builtin(name, args[0], expr);
}

/// IREP2 form of clang_c_adjust::adjust_address_of's array decay: `&a` on an
/// array is `&a[0]`, and the pointer's subtype follows the element.
///
/// The conditional distribution the legacy arm also does -- `&(c ? a : b)` into
/// `c ? &a : &b`, which #6291 needs for the pointer analysis to resolve either
/// arm -- is not ported: no corpus input reaches it under this flag, and an arm
/// no test executes is the trap §90.4 records.
void clang_c_adjust_irep2::adjust_address_of(expr2tc &expr)
{
  const address_of2t &a = to_address_of2t(expr);
  if (is_nil_expr(a.ptr_obj))
    return;

  const type2tc obj_type = ns.follow(a.ptr_obj->type);
  if (!is_array_type(obj_type))
    return;

  const type2tc &elem = to_array_type(obj_type).subtype;
  const expr2tc idx =
    index2tc(elem, a.ptr_obj, gen_zero(migrate_type(index_type())));
  expr = address_of2tc(elem, idx, a.implicit);
}

/// IREP2 form of clang_c_adjust::adjust_struct's insertion loop. A struct
/// literal reaches this pass with an operand per *declared* member, while
/// add_padding has given the type its synthetic ones; unpadded, the value is
/// short a component and a downstream member read resolves by position onto the
/// wrong field. pad_struct_operands is the shared helper the Python adjuster
/// already uses for the same job (irep2/irep2_utils.h).
void clang_c_adjust_irep2::adjust_struct(expr2tc &expr)
{
  // The literal's own type is an inline copy the converter recorded before
  // add_padding ran, so ns.follow leaves it short the synthetic members. The
  // padded layout lives on the tag symbol; resolve by name to reach it. Legacy
  // adjust_struct needs no such step -- there the value's type is a
  // symbol_typet, which follow resolves for it.
  const type2tc t = ns.follow(expr->type);
  if (!is_struct_type(t))
    return;

  const symbolt *tag =
    context.find_symbol("tag-" + to_struct_type(t).name.as_string());
  if (tag == nullptr || !tag->is_type)
    return;

  const type2tc padded = migrate_type(tag->get_type());
  if (!is_struct_type(padded))
    return;

  const struct_type2t &st = to_struct_type(padded);
  std::vector<expr2tc> ops = to_constant_struct2t(expr).datatype_members;
  if (ops.size() == st.members.size())
    return;

  ops = pad_struct_operands(st, ops);
  // A residual mismatch is not this pass's to guess at: leave the literal as
  // it stands rather than build one the type cannot describe.
  if (ops.size() == st.members.size())
    expr = constant_struct2tc(padded, ops);
}

/// adjust_struct retypes an element to its tag's padded layout, which leaves
/// an enclosing array literal still naming the unpadded element type. The
/// operands are walked before the node itself, so the retyped elements are
/// already in place here; value_sett::assign's base_type_eq rejects the pair
/// otherwise.
void clang_c_adjust_irep2::adjust_array_subtype(expr2tc &expr)
{
  const constant_array2t &a = to_constant_array2t(expr);
  if (a.datatype_members.empty())
    return;

  const array_type2t &at = to_array_type(expr->type);
  const type2tc &elem = a.datatype_members.front()->type;
  if (
    !is_struct_type(at.subtype) || !is_struct_type(elem) || at.subtype == elem)
    return;

  expr = constant_array2tc(
    array_type2tc(elem, at.array_size, at.size_is_infinite),
    a.datatype_members);
}

/// IREP2 form of clang_c_adjust::adjust_decl's trailing `gen_typecast`: a
/// declaration's initialiser converts to the declared type. Distinct from
/// adjust_plain_assignment, which handles the *expression* form
/// (`sideeffect_assign2t`). Without it a narrower declared type keeps the
/// promoted operand type -- `_ExtInt(10) z = x + y` initialises from an `int`
/// -- and the solver is handed mismatching sorts.
void clang_c_adjust_irep2::adjust_decl_init(expr2tc &expr)
{
  const code_decl2t &d = to_code_decl2t(expr);
  if (is_nil_expr(d.init))
    return;

  expr2tc init = d.init;
  c_implicit_typecast(init, expr->type, ns);

  if (init != d.init)
    expr = code_decl2tc(expr->type, d.value, init, d.location);
}

void clang_c_adjust_irep2::hoist_for_init(expr2tc &expr)
{
  const code_for2t &f = to_code_for2t(expr);
  if (is_nil_expr(f.init))
    return;

  // A default-constructed locationt is empty but not nil, and migrate_expr_back
  // guards only on nil: left default it reaches convert_block, which stamps it
  // on every destructor it unwinds. Same idiom as goto_convert_functions.cpp.
  locationt end_location;
  if (!is_nil_expr(f.body) && is_code_block2t(f.body))
    end_location = to_code_block2t(f.body).end_location;
  else
    end_location.make_nil();

  // pragma_unroll_count is excluded from code_for2t::fields, so it does not
  // participate in equality and a rebuild that drops it compares equal to one
  // that keeps it. Every loop rebuild in this pass has to carry it explicitly.
  const expr2tc bare = code_for2tc(
    expr2tc(), f.cond, f.iter, f.body, f.location, f.pragma_unroll_count);

  // Splice a block-shaped init rather than nesting it: an inner block would end
  // the declaration's scope at its own closing brace, so the variable would be
  // DEAD before the loop that reads it. clang_c_adjust moves the init operand
  // itself, which is why the legacy hoist puts the declaration directly in the
  // wrapper.
  std::vector<expr2tc> ops;
  if (is_code_block2t(f.init))
    for (const expr2tc &op : to_code_block2t(f.init).operands)
      ops.push_back(op);
  else
    ops.push_back(f.init);
  ops.push_back(bare);

  expr = code_block2tc(ops, f.location, end_location);
}

void clang_c_adjust_irep2::adjust_binary_arith_operands(expr2tc &expr)
{
  expr2tc op0 = *expr->get_sub_expr(0);
  expr2tc op1 = *expr->get_sub_expr(1);
  if (is_nil_expr(op0) || is_nil_expr(op1))
    return;

  // A complex operand is adjust_complex_arith's, and it decomposes the node
  // rather than converting it.
  if (is_complex_type(op0->type) || is_complex_type(op1->type))
    return;

  const expr2tc before0 = op0, before1 = op1;
  c_implicit_typecast_arithmetic(op0, op1, ns);

  if (op0 != before0 || op1 != before1)
  {
    unsigned i = 0;
    expr->Foreach_operand(
      [&i, &op0, &op1](expr2tc &o) { o = i++ ? op1 : op0; });
  }

  // adjust_expr_binary_arithmetic re-types the node once the operands agree.
  // Not folded into the branch above: the operands can already agree with each
  // other and still disagree with the node.
  if (
    op0->type == op1->type && is_number_type(op0->type) &&
    expr->type != op0->type)
    expr = expr->with_type(op0->type);
}

/// IREP2 form of clang_c_adjust::adjust_side_effect_assignment's "assign" case:
/// the node takes the target's type and the source converts to it. The compound
/// operators ("assign+", ...) are a larger arm carrying a complex lowering of
/// their own, and are left where this mode already had them.
void clang_c_adjust_irep2::adjust_plain_assignment(expr2tc &expr)
{
  const sideeffect_assign2t &a = to_sideeffect_assign2t(expr);
  if (a.op != "assign" || is_nil_expr(a.lhs) || is_nil_expr(a.rhs))
    return;

  const type2tc target = a.lhs->type;
  expr2tc rhs = a.rhs;
  c_implicit_typecast(rhs, target, ns);

  if (rhs != a.rhs || expr->type != target)
    expr = sideeffect_assign2tc(target, a.op, a.lhs, rhs, a.location);
}

/// The shift spellings clang_c_adjust returns early on: it promotes only the
/// right operand there, which the corpus shows is already the migrated shape.
static bool is_shift_assignment(const irep_idt &op)
{
  return op == "assign_shl" || op == "assign_shr" || op == "assign_lshr" ||
         op == "assign_ashr";
}

void clang_c_adjust_irep2::adjust_compound_assignment(expr2tc &expr)
{
  const sideeffect_assign2t &a = to_sideeffect_assign2t(expr);
  if (a.op == "assign" || is_shift_assignment(a.op))
    return;
  if (is_nil_expr(a.lhs) || is_nil_expr(a.rhs))
    return;

  if (is_complex_type(a.lhs->type) || is_complex_type(a.rhs->type))
  {
    lower_complex_compound_assignment(expr);
    return;
  }

  const type2tc target = a.lhs->type;
  expr2tc lhs = a.lhs, rhs = a.rhs;
  c_implicit_typecast_arithmetic(lhs, rhs, ns);

  if (lhs != a.lhs || rhs != a.rhs || expr->type != target)
    expr = sideeffect_assign2tc(target, a.op, lhs, rhs, a.location);
}

/// IREP2 form of the `gen_typecast_bool` each of adjust_ifthenelse,
/// adjust_while and adjust_for applies to its controlling expression.
/// goto_convert's branch lowering rejects a non-boolean guard, so this is the
/// statement-level counterpart of adjust_if_expr's operand half.
void clang_c_adjust_irep2::adjust_statement_condition(expr2tc &expr)
{
  expr2tc cond;
  if (is_code_ifthenelse2t(expr))
    cond = to_code_ifthenelse2t(expr).cond;
  else if (is_code_while2t(expr))
    cond = to_code_while2t(expr).cond;
  else if (is_code_dowhile2t(expr))
    cond = to_code_dowhile2t(expr).cond;
  else
    cond = to_code_for2t(expr).cond;

  // A `for` may omit its condition entirely.
  if (is_nil_expr(cond) || is_bool_type(cond->type))
    return;

  const expr2tc before = cond;
  c_implicit_typecast(cond, get_bool_type(), ns);
  if (cond == before)
    return;

  if (is_code_ifthenelse2t(expr))
  {
    const code_ifthenelse2t &i = to_code_ifthenelse2t(expr);
    expr = code_ifthenelse2tc(cond, i.then_case, i.else_case, i.location);
  }
  else if (is_code_while2t(expr))
  {
    const code_while2t &w = to_code_while2t(expr);
    expr = code_while2tc(cond, w.body, w.location, w.pragma_unroll_count);
  }
  else if (is_code_dowhile2t(expr))
  {
    const code_dowhile2t &w = to_code_dowhile2t(expr);
    expr = code_dowhile2tc(cond, w.body, w.location, w.pragma_unroll_count);
  }
  else
  {
    const code_for2t &f = to_code_for2t(expr);
    expr = code_for2tc(
      f.init, cond, f.iter, f.body, f.location, f.pragma_unroll_count);
  }
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

/// A function designator used as a value is sugar for `&f`
/// (clang_c_adjust::adjust_symbol). Applied from the parent rather than at the
/// symbol itself: `address_of2t` asserts its operand is not another address_of,
/// so a user-written `&f` must not be wrapped again -- where the legacy pass
/// builds `&(&f)` and collapses it in adjust_address_of, this never builds it.
void clang_c_adjust_irep2::adjust_function_designators(expr2tc &expr)
{
  if (is_address_of2t(expr))
    return;

  expr->Foreach_operand([](expr2tc &op) {
    if (!is_nil_expr(op) && is_symbol2t(op) && is_code_type(op->type))
      op = address_of2tc(op->type, op, true);
  });
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

  if (is_nil_expr(callee))
    return;

  // `f(x)` arrives as a call through the &f sugar adjust_symbol inserted; strip
  // it back off so goto_convert sees a direct call. A user-written `(&f)(x)`
  // carries the same shape and is told apart only by the implicit bit (§100).
  if (is_address_of2t(callee) && to_address_of2t(callee).implicit)
  {
    const expr2tc target = to_address_of2t(callee).ptr_obj;
    if (is_code_function_call2t(expr))
      to_code_function_call2t(expr).function = target;
    else
      to_sideeffect2t(expr).operand = target;
    return;
  }

  if (!is_pointer_type(callee->type))
    return;

  const expr2tc deref =
    dereference2tc(to_pointer_type(callee->type).subtype, callee);

  if (is_code_function_call2t(expr))
    to_code_function_call2t(expr).function = deref;
  else
    to_sideeffect2t(expr).operand = deref;
}

/// True when \p arg is bound to parameter \p i of \p callee rather than
/// converted to its type -- `va_start(ap, n)` hands the callee `&ap`.
///
/// `pointer_type2t` has no field for the `#reference` bit
/// `clang_c_convertert::get_type` sets, so `migrate_type` drops it and the rule
/// `c_typecastt::implicit_typecast_followed` applies on the legacy path cannot
/// be stated against the IREP2 types. It survives on the symbol's legacy
/// `typet`, which is what this reads -- the same route the implicit-callee arm
/// above already takes to recover a base name. The rest of the conjunction is
/// legacy's own precondition, kept whole so the two paths decide alike.
///
/// Whether a parameter is a reference is a property of the target, not of the
/// callee's name: clang's `A` builtin-type code is an lvalue reference where
/// `__builtin_va_list` is a pointer or a struct, and an already-decayed
/// pointer where it is an array (x86-64 Linux). Reading the bit follows the
/// target; a name test would take the address on both.
static bool binds_by_reference(
  const expr2tc &callee,
  const expr2tc &arg,
  const type2tc &param,
  std::size_t i,
  const contextt &context,
  const namespacet &ns)
{
  // address_of2t asserts its operand is not another address_of, so a caller
  // that already took the address is left alone.
  if (is_address_of2t(arg) || !is_pointer_type(param) || arg->type == param)
    return false;

  if (ns.follow(to_pointer_type(param).subtype)->type_id != arg->type->type_id)
    return false;

  if (!is_symbol2t(callee))
    return false;

  const symbolt *s = context.find_symbol(to_symbol2t(callee).thename);
  if (s == nullptr || !s->get_type().is_code())
    return false;

  const code_typet::argumentst &decl = to_code_type(s->get_type()).arguments();
  return i < decl.size() && is_lvalue_or_rvalue_reference(decl[i].type());
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
    {
      // Two function-pointer types differing only in argument_names denote the
      // same type (C11 6.7.6.3p15); casting between them is a divergence, not a
      // conversion (§100.1).
      if (same_function_pointer_ignoring_argument_names(arg->type, params[i]))
        continue;

      // Converted instead of bound, `va_start` gets the va_list's own value
      // and the callee initialises whatever that value happens to point at.
      if (binds_by_reference(callee, arg, params[i], i, context, ns))
      {
        arg = address_of2tc(arg->type, arg);
        continue;
      }

      c_implicit_typecast(arg, params[i], ns);
    }
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

/// An expression statement whose value has array type -- `y->ss;` where `y`
/// points at a struct with an array member -- is rewritten to `&y->ss[0]`.
/// clang_c_adjust does this because the dereference code does not assume such
/// an object exists; the statement's value is unused, so taking the first
/// element's address is free (clang_c_adjust_code.cpp:57-74).
///
/// An assignment operand is exempt there and here: the array is the assignment
/// target, not a value being discarded.
void clang_c_adjust_irep2::adjust_expression_statement(expr2tc &expr)
{
  const code_expression2t &stmt = to_code_expression2t(expr);
  const expr2tc &op = stmt.operand;
  if (is_nil_expr(op) || is_sideeffect_assign2t(op) || is_code_assign2t(op))
    return;

  const type2tc t = ns.follow(op->type);
  if (!is_array_type(t) && !is_vector_type(t))
    return;

  const type2tc &elem =
    is_array_type(t) ? to_array_type(t).subtype : to_vector_type(t).subtype;
  expr = code_expression2tc(
    address_of2tc(
      elem, index2tc(elem, op, gen_zero(migrate_type(index_type())))),
    stmt.location);
}

/// Dereferencing a pointer to a function yields a function designator, which
/// converts straight back to a pointer (C11 6.3.2.1p4) -- so `*f` is `f`, and
/// `******f` too. clang_c_adjust::adjust_dereference re-takes the address for
/// exactly this case; left bare, the code-typed dereference reaches a consumer
/// that wants a pointer.
///
/// Only that arm is ported: the array and pointer-subtype arms above it
/// retype a node the migration already builds with the right type, so no
/// corpus input distinguishes them.
void clang_c_adjust_irep2::adjust_dereference(expr2tc &expr)
{
  if (!is_code_type(expr->type))
    return;

  expr = address_of2tc(expr->type, expr, true);
}

/// IREP2 form of clang_c_adjust::lower_complex_compound_assignment. `a op= b`
/// over a complex operand becomes `a = a op b`, with the binary node handed to
/// adjust_complex_arith for the component-level decomposition. goto_convert's
/// remove_assignment rebuilds the compound form long after adjustment, so a
/// node left here reaches the SMT layer as a raw complex operator and the
/// backend faults on it (#6713).
void clang_c_adjust_irep2::lower_complex_compound_assignment(expr2tc &expr)
{
  const sideeffect_assign2t &a = to_sideeffect_assign2t(expr);

  // adjust_complex_arith reads each operand twice, once per component, so a
  // side-effecting target would be evaluated twice. Same decline as there.
  if (contains_sideeffect(a.lhs))
    return;

  const type2tc &ct = a.lhs->type;

  // `a *= 2.0f` leaves the scalar as-is, and the arithmetic node's consistency
  // check rejects an operand narrower than its type before adjust_complex_arith
  // gets to promote it. Promote here, the same way that function does.
  expr2tc rhs = a.rhs;
  if (!is_complex_type(rhs->type))
    rhs = constant_struct2tc(
      ct, std::vector<expr2tc>{rhs, gen_zero(to_complex_type(ct).subtype)});

  expr2tc binop;
  if (a.op == "assign+")
    binop = add2tc(ct, a.lhs, rhs);
  else if (a.op == "assign-")
    binop = sub2tc(ct, a.lhs, rhs);
  else if (a.op == "assign*")
    binop = mul2tc(ct, a.lhs, rhs);
  else if (a.op == "assign_div")
    binop = div2tc(ct, a.lhs, rhs);
  else
    return;

  const expr2tc before = binop;
  adjust_complex_arith(binop);
  // It declines on a side-effecting operand; leave the node rather than emit a
  // plain assignment of an undecomposed complex operator.
  if (binop == before)
    return;

  expr = sideeffect_assign2tc(ct, "assign", a.lhs, binop, a.location);
}

/// clang emits `ieee_*` for scalar float arithmetic itself, but hands over a
/// plain `+`/`-`/`*`/`/` when the operands are vectors of float and leaves
/// clang_c_adjust::adjust_float_arith to promote it. Unpromoted, the backend is
/// handed a bitvector operator over a floating-point vector and aborts.
///
/// The legacy arm returns before attaching a rounding mode for a vector ("BUG:
/// setting rounding_mode breaks migration"), and migrate_rounding_mode then
/// synthesises the default symbol for the attribute-less node -- so the node
/// the default path produces carries that symbol, and this builds the same one.
void clang_c_adjust_irep2::adjust_vector_float_arith(expr2tc &expr)
{
  const type2tc t = ns.follow(expr->type);
  if (!is_vector_type(t) || !is_floatbv_type(to_vector_type(t).subtype))
    return;

  const expr2tc &l = *expr->get_sub_expr(0);
  const expr2tc &r = *expr->get_sub_expr(1);
  if (is_nil_expr(l) || is_nil_expr(r))
    return;

  const expr2tc rm = symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");

  if (is_add2t(expr))
    expr = ieee_add2tc(expr->type, l, r, rm);
  else if (is_sub2t(expr))
    expr = ieee_sub2tc(expr->type, l, r, rm);
  else if (is_mul2t(expr))
    expr = ieee_mul2tc(expr->type, l, r, rm);
  else if (is_div2t(expr))
    expr = ieee_div2tc(expr->type, l, r, rm);
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

/// C11 6.5.3.3: the operand of unary `-` and `~` undergoes integer promotion,
/// so a boolean one -- a comparison, `||` or `&&` -- becomes int. Left boolean
/// it reaches the solver where a bitvector is wanted (#4078).
void clang_c_adjust_irep2::promote_unary_bool_operand(expr2tc &expr)
{
  const expr2tc &op = *expr->get_sub_expr(0);
  if (is_nil_expr(op) || !is_bool_type(op->type) || is_bool_type(expr->type))
    return;

  expr2tc promoted = op;
  c_implicit_typecast(promoted, expr->type, ns);
  if (promoted == op)
    return;

  expr = is_neg2t(expr) ? expr2tc(neg2tc(expr->type, promoted))
                        : expr2tc(bitnot2tc(expr->type, promoted));
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

void clang_c_adjust_irep2::declare_polymorphic_builtin(expr2tc &expr)
{
  // The caller admits these two kinds only, so there is no third arm to guard:
  // to_sideeffect2t throws loudly on anything else rather than silently
  // skipping a call that should have been declared.
  expr2tc callee;
  const std::vector<expr2tc> *args = nullptr;
  locationt loc;
  if (is_code_function_call2t(expr))
  {
    const code_function_call2t &call = to_code_function_call2t(expr);
    callee = call.function;
    args = &call.operands;
    loc = call.location;
  }
  else
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    if (se.kind != sideeffect_allockind::function_call)
      return;
    callee = se.operand;
    args = &se.arguments;
    loc = enclosing_location;
  }

  if (is_nil_expr(callee) || !is_symbol2t(callee))
    return;

  // Every arm of the matcher selects on the first argument's *type* alone, so
  // the values need not cross the seam. A future arm that reads a value gets a
  // nil operand and fails visibly rather than silently selecting wrong.
  exprt::operandst arg_types;
  arg_types.reserve(args->size());
  for (const expr2tc &arg : *args)
  {
    if (is_nil_expr(arg))
      return;
    arg_types.emplace_back(exprt("nil", migrate_type_back(arg->type)));
  }

  const irep_idt id = to_symbol2t(callee).thename;

  symbol_exprt legacy_callee(id, migrate_type_back(callee->type));
  legacy_callee.name(get_pretty_name(id2string(id)));
  legacy_callee.location() = loc;

  const exprt poly = clang_c_adjust::declare_gcc_polymorphic_builtin(
    legacy_callee, arg_types, loc, context);
  if (poly.is_nil())
    return;

  expr2tc target;
  migrate_expr(poly, target);
  if (is_code_function_call2t(expr))
    to_code_function_call2t(expr).function = target;
  else
    to_sideeffect2t(expr).operand = target;
}

void clang_c_adjust_irep2::declare_implicit_callee(
  const expr2tc &expr,
  const locationt &stmt_location)
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
  else if (is_sideeffect2t(expr))
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    if (se.kind != sideeffect_allockind::function_call)
      return;
    callee = se.operand;
    // sideeffect2t has no location of its own. The enclosing statement's is
    // the call's only when the call is the whole statement, which is the one
    // position this is passed from.
    loc = stmt_location;
  }
  else
    return;

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
