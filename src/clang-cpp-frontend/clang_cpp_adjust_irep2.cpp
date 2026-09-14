#include <clang-cpp-frontend/clang_cpp_adjust_irep2.h>
#include <clang-cpp-frontend/clang_cpp_code_gen.h>
#include <clang-cpp-frontend/clang_cpp_destructor_call.h>
#include <goto-programs/destructor.h>
#include <util/irep/std_code.h>
#include <clang-cpp-frontend/clang_cpp_exception_id.h>

/// The guards live on the C pass's translation unit as file-local statics, so
/// they are re-declared here rather than shared: they read only the node, and a
/// second definition of a one-line predicate is cheaper than exporting them.
#include <clang-c-frontend/clang_c_adjust_guards.h>

/// A member whose type is code: a method named through `.` or `->`. A data
/// member is left alone, and so is a member with no component name.
static bool is_cpp_member_call(const expr2tc &expr)
{
  return is_member2t(expr) && is_code_type(expr->type) &&
         !to_member2t(expr).member.empty();
}

/// A source-level try/catch whose handler ids have not been computed yet. The
/// post-goto-convert CATCH marker carries no operands and is left alone.
static bool is_unresolved_cpp_catch(const expr2tc &expr)
{
  return is_code_cpp_catch2t(expr) &&
         to_code_cpp_catch2t(expr).operands.size() > 1 &&
         to_code_cpp_catch2t(expr).exception_list.empty();
}

/// A `delete` whose destructor call has not been attached yet.
static bool is_unresolved_cpp_delete(const expr2tc &expr)
{
  if (!is_sideeffect2t(expr))
    return false;

  const sideeffect2t &se = to_sideeffect2t(expr);
  if (
    se.kind != sideeffect2t::allockind::cpp_delete &&
    se.kind != sideeffect2t::allockind::cpp_delete_array)
    return false;

  // arguments[0] is the destructor call, [1] a replaced operator delete. A
  // delete that has only the latter still arrives with a nil in [0]
  // (migrate.cpp pads), so "no arguments" is not the same question as "no
  // destructor yet" (github #6494).
  return se.arguments.empty() || is_nil_expr(se.arguments[0]);
}

static bool is_unresolved_cpp_throw(const expr2tc &expr)
{
  return is_code_cpp_throw2t(expr) &&
         to_code_cpp_throw2t(expr).exception_list.empty() &&
         !is_nil_expr(to_code_cpp_throw2t(expr).operand);
}

#define ARM(member)                                                            \
#  member,                                                                     \
    +[](clang_cpp_adjust_irep2 & self, expr2tc & expr) { self.member(expr); }

/// Inherited arms only, in the C pass's order. What C++ adds goes here as the
/// divergence census names it (scope-clang-cpp-irep2.md §3.1).
const clang_cpp_adjust_irep2::arm clang_cpp_adjust_irep2::arms[] = {
  {ARM(adjust_cpp_catch), is_unresolved_cpp_catch},
  {ARM(adjust_cpp_throw), is_unresolved_cpp_throw},
  {ARM(adjust_cpp_delete), is_unresolved_cpp_delete},
  {ARM(adjust_cpp_member), is_cpp_member_call},
  {ARM(adjust_function_designators), nullptr},
  {ARM(adjust_boolean_operands), is_short_circuit},
  {ARM(adjust_call_callee), is_call_site},
  {ARM(adjust_call_signature), is_call_site},
  {ARM(adjust_call_arguments), is_call_site},
  {ARM(adjust_if_expr), is_if2t},
  {ARM(adjust_complex_arith), is_binary_arith},
  {ARM(adjust_vector_float_arith), is_binary_arith},
  {ARM(adjust_statement_condition), is_statement_with_condition},
  {ARM(hoist_for_init), is_code_for2t},
  {ARM(adjust_expression_statement), is_code_expression2t},
  {ARM(adjust_comma_type), is_code_comma2t},
  {ARM(adjust_struct), is_constant_struct2t},
  {ARM(adjust_array_subtype), is_constant_array2t},
  {ARM(adjust_decl_init), is_code_decl2t},
  {ARM(adjust_ptr_mem), is_ptr_mem2t},
  {ARM(adjust_dereference), is_dereference2t},
  {ARM(adjust_complex_unary), is_complex_unary},
  {ARM(promote_unary_bool_operand), is_promotable_unary},
  {ARM(adjust_relational), is_relational},
  {ARM(adjust_increment_reference), is_increment_sideeffect},
  {ARM(adjust_special_functions), is_sideeffect2t},
  {ARM(adjust_binary_arith_operands), is_arith_or_bitwise},
  {ARM(adjust_shift_operands), is_shift},
  {ARM(fan_out_array_construction), is_code_block2t},
  {ARM(adjust_plain_assignment), is_sideeffect_assign2t},
  {ARM(adjust_compound_assignment), is_sideeffect_assign2t},
  {ARM(adjust_derived_to_base), is_derived_to_base_cast},
  {ARM(adjust_base_to_derived), is_base_to_derived_cast},
  {ARM(adjust_address_of), is_address_of2t},
};

#undef ARM

std::vector<clang_cpp_adjust_irep2::arm_info>
clang_cpp_adjust_irep2::arm_order()
{
  std::vector<arm_info> order;
  order.reserve(std::size(arms));
  for (const arm &a : arms)
    order.push_back({a.name, a.when});
  return order;
}

void clang_cpp_adjust_irep2::adjust_sole_arms(expr2tc &expr)
{
  run_adjust_arms(*this, arms, expr);
}

bool clang_cpp_adjust_irep2::is_constructor_call(const expr2tc &call)
{
  if (!is_sideeffect2t(call))
    return false;

  const sideeffect2t &se = to_sideeffect2t(call);
  if (
    se.kind != sideeffect_allockind::function_call || is_nil_expr(se.operand) ||
    !is_symbol2t(se.operand))
    return false;

  const symbolt *s = context.find_symbol(to_symbol2t(se.operand).thename);
  return s != nullptr && s->get_type().is_code() &&
         to_code_type(s->get_type()).return_type().id() == "constructor";
}

expr2tc clang_cpp_adjust_irep2::find_constructor_call(const expr2tc &e)
{
  if (is_nil_expr(e))
    return expr2tc();
  if (is_constructor_call(e))
    return e;

  expr2tc found;
  e->foreach_operand([this, &found](const expr2tc &op) {
    if (is_nil_expr(found))
      found = find_constructor_call(op);
  });
  return found;
}

expr2tc clang_cpp_adjust_irep2::array_decl_constructor(const expr2tc &stmt)
{
  if (!is_code_decl2t(stmt))
    return expr2tc();

  const code_decl2t &d = to_code_decl2t(stmt);
  if (is_nil_expr(d.init) || !is_array_type(ns.follow(d.type)))
    return expr2tc();

  // A function-local static is constructed by static_lifetime_init, not from
  // the body; expanding here would construct it again on every call. That half
  // is clang_cpp_maint::adjust_init's (clang_cpp_main.cpp), which keys on the
  // `#constructor` marker this pass's write-back destroys -- a static or global
  // class-typed array is unconstructed under this flag, §3.16's open row.
  const symbolt *s = context.find_symbol(d.value);
  if (s == nullptr || s->static_lifetime)
    return expr2tc();

  // Whole-array default/value construction only: the initialiser is a *single*
  // constructor call whose type is the array. `B a[2] = {B(1), B(2)}` arrives
  // as a constant_array of per-element initialisers, and fanning that out would
  // construct every element with element 0's arguments.
  if (
    !is_sideeffect2t(d.init) ||
    (to_sideeffect2t(d.init).kind != sideeffect_allockind::temporary_object &&
     !is_constructor_call(d.init)))
    return expr2tc();

  const expr2tc ctor = find_constructor_call(d.init);
  if (is_nil_expr(ctor) || to_sideeffect2t(ctor).arguments.empty())
    return expr2tc();

  return ctor;
}

bool clang_cpp_adjust_irep2::construct_elements(
  const expr2tc &array,
  const expr2tc &ctor,
  std::vector<expr2tc> &out)
{
  const type2tc array_type = ns.follow(array->type);
  const array_type2t &at = to_array_type(array_type);
  // Legacy aborts here ("cannot determine array size for local ctor init").
  // Declining instead leaves the declaration as it arrived: the caller commits
  // nothing until this returns true, so a size it cannot read never costs the
  // object its initialiser.
  if (!is_constant_int2t(at.array_size))
    return false;

  const sideeffect2t &call = to_sideeffect2t(ctor);
  const BigInt count = to_constant_int2t(at.array_size).value;
  for (BigInt i = 0; i < count; ++i)
  {
    const expr2tc element =
      index2tc(at.subtype, array, constant_int2tc(index_type2(), i));
    if (is_array_type(ns.follow(at.subtype)))
    {
      if (!construct_elements(element, ctor, out))
        return false;
      continue;
    }

    // arguments[0] is the object argument; array_decl_constructor declines a
    // call that has none. The element's type, not the initialiser's: see the
    // fold's own choice below.
    std::vector<expr2tc> args = call.arguments;
    args[0] = address_of2tc(at.subtype, element);
    out.push_back(code_expression2tc(
      sideeffect2tc(
        at.subtype,
        call.operand,
        call.size,
        args,
        call.alloctype,
        call.kind,
        call.location,
        call.constructor),
      call.location));
  }

  return true;
}

void clang_cpp_adjust_irep2::fan_out_array_construction(expr2tc &expr)
{
  const code_block2t &b = to_code_block2t(expr);
  std::vector<expr2tc> out;
  bool fanned = false;

  for (const expr2tc &stmt : b.operands)
  {
    const expr2tc ctor = array_decl_constructor(stmt);
    if (is_nil_expr(ctor))
    {
      out.push_back(stmt);
      continue;
    }

    const code_decl2t &d = to_code_decl2t(stmt);
    std::vector<expr2tc> calls;
    if (!construct_elements(symbol2tc(d.type, d.value), ctor, calls))
    {
      // Legacy aborts on this input, so nothing downstream expects a partly
      // constructed array: every element past the first is left
      // nondeterministic, which surfaces as a violated assertion rather than
      // as the decline it is.
      const symbolt *sym = context.find_symbol(d.value);
      log_warning(
        "{}: '{}' has a non-constant extent, so only its first element is "
        "constructed",
        d.location.as_string(),
        sym != nullptr ? sym->name : d.value);
      out.push_back(stmt);
      continue;
    }

    fanned = true;
    out.push_back(code_decl2tc(d.type, d.value, expr2tc(), d.location));
    out.insert(out.end(), calls.begin(), calls.end());
  }

  if (fanned)
    expr = code_block2tc(out, b.location, b.end_location);
}

void clang_cpp_adjust_irep2::adjust_before_operands(expr2tc &expr)
{
  if (is_sideeffect_assign2t(expr))
    fold_constructor_assignment(expr);
}

void clang_cpp_adjust_irep2::fold_constructor_assignment(expr2tc &expr)
{
  expr2tc folded;
  {
    const sideeffect_assign2t &a = to_sideeffect_assign2t(expr);
    if (a.op != "assign" || is_nil_expr(a.lhs) || is_nil_expr(a.rhs))
      return;

    if (!is_constructor_call(a.rhs))
      return;

    const sideeffect2t &call = to_sideeffect2t(a.rhs);

    std::vector<expr2tc> args = call.arguments;
    args.insert(args.begin(), address_of2tc(a.lhs->type, a.lhs));

    // The call's value is the object constructed, so it takes the object's
    // type, not the initialiser's: for a class-typed array member the
    // converter hands each per-element call the whole array's type, and left
    // there adjust_expression_statement reads it as an array-valued statement
    // and wraps it in `&stmt[0]`.
    folded = sideeffect2tc(
      a.lhs->type,
      call.operand,
      call.size,
      args,
      call.alloctype,
      call.kind,
      call.location,
      call.constructor);
  }

  // The views above are dead here on purpose: this drops the assignment node
  // they referenced.
  expr = folded;
}

void clang_cpp_adjust_irep2::align_call_return_type(
  expr2tc &expr,
  const symbolt &callee)
{
  if (!is_sideeffect2t(expr))
    return;

  // Read the return type in legacy form: "constructor" is an irept id with no
  // IREP2 spelling to test against.
  const typet &ret = to_code_type(callee.get_type()).return_type();
  if (ret.is_nil() || ret.id() == "constructor")
    return;

  const type2tc ret2 = migrate_type(ret);
  if (expr->type == ret2)
    return;

  const sideeffect2t &se = to_sideeffect2t(expr);
  expr = sideeffect2tc(
    ret2,
    se.operand,
    se.size,
    se.arguments,
    se.alloctype,
    se.kind,
    se.location);
}

void clang_cpp_adjust_irep2::adjust_cpp_member(expr2tc &expr)
{
  const member2t &m = to_member2t(expr);
  const symbolt *comp = ns.lookup(m.member);
  if (!comp)
  {
    // The legacy arm aborts here. Declining instead would hand goto_convert a
    // member callee it rejects, so the diagnostic is worth more than the
    // fall-through -- but it names the member, which the legacy message does.
    log_error(
      "adjust_cpp_member: unresolved C++ member component `{}` (source type "
      "id `{}`)",
      m.member,
      get_type_id(m.source_value->type));
    abort();
  }

  assert(comp->get_type().is_code());
  expr = symbol2tc(migrate_type(comp->get_type()), comp->id);
}

void clang_cpp_adjust_irep2::adjust_cpp_catch(expr2tc &expr)
{
  const code_cpp_catch2t &c = to_code_cpp_catch2t(expr);

  // One id per handler, parallel to operands[1..N]; the legacy arm keeps only
  // the leading id per handler and expands base classes at the throw site.
  std::vector<irep_idt> ids;
  for (std::size_t i = 1; i < c.operands.size(); i++)
  {
    // is_catch stays false, as both legacy call sites leave it: it is what
    // strips the `tag-` prefix, and a throw's ids are stripped too, so a
    // handler id of `tag-E` would match no throw.
    std::vector<irep_idt> one;
    convert_exception_id(ns, migrate_type_back(c.operands[i]->type), "", one);
    ids.push_back(one.empty() ? irep_idt() : one.front());
  }

  expr = code_cpp_catch2tc(ids, c.operands, c.location);
}

void clang_cpp_adjust_irep2::adjust_cpp_throw(expr2tc &expr)
{
  const code_cpp_throw2t &th = to_code_cpp_throw2t(expr);

  // Every id the thrown type resolves to, most derived first, so a handler for
  // a base catches it.
  std::vector<irep_idt> ids;
  convert_exception_id(ns, migrate_type_back(th.operand->type), "", ids);

  expr = code_cpp_throw2tc(th.operand, ids, th.location);
}

void clang_cpp_adjust_irep2::adjust_cpp_delete(expr2tc &expr)
{
  const sideeffect2t &se = to_sideeffect2t(expr);

  const typet deleted = migrate_type_back(se.type);
  const struct_typet *class_type = resolve_class_type(ns, deleted);
  if (!class_type)
    return;

  const struct_typet::componentt *dtor =
    get_destructor_component(ns, *class_type);
  if (!dtor)
    return;

  // The legacy arm builds this in the old representation and the seam carries
  // it; building it the same way keeps one definition of what `delete` calls.
  const exprt new_object("new_object", deleted);
  code_function_callt destructor;
  destructor.function() =
    destructor_binding(ns, *class_type, *dtor, new_object);
  destructor.arguments().push_back(address_of_exprt(new_object));

  expr2tc call;
  migrate_expr(destructor, call);

  // Fill the destructor slot without disturbing a replaced operator delete
  // sitting behind it.
  std::vector<expr2tc> args = se.arguments;
  if (args.empty())
    args.push_back(call);
  else
    args[0] = call;

  expr = sideeffect2tc(
    se.type, se.operand, se.size, args, se.alloctype, se.kind, se.location);
}

namespace
{
/// `r` used as a value is `*r`. A cast of one is handled first, so `(int)r`
/// becomes `(int)*r` rather than a cast of the pointer.
/// The referent's type, resolved. A dereference2t left with a by-name tag has
/// no width or alignment, and symex reports that as a spurious alignment
/// failure rather than as the unresolved type it is.
type2tc referent_type(const namespacet &ns, const type2tc &ref)
{
  return ns.follow(to_pointer_type(ref).subtype);
}

void convert_reference(const namespacet &ns, expr2tc &expr)
{
  // Unexercised over the whole C++ corpus (0 hits in 1353 tests): a reference
  // *symbol* under a cast does not occur, because get_decl_ref dereferences a
  // reference variable at conversion time. Kept because clang_cpp_adjust's
  // convert_reference carries the identical guard, and a port that prunes a
  // branch the original has is harder to compare against it later.
  if (is_typecast2t(expr))
  {
    const typecast2t cast = to_typecast2t(expr);
    if (is_symbol2t(cast.from) && is_reference_type(cast.from->type))
      expr = typecast2tc(
        cast.type,
        dereference2tc(referent_type(ns, cast.from->type), cast.from),
        cast.rounding_mode,
        cast.derived_to_base,
        cast.base_to_derived);
  }

  if (is_reference_type(expr->type))
    expr = dereference2tc(referent_type(ns, expr->type), expr);
}
} // namespace

void clang_cpp_adjust_irep2::adjust_reference(expr2tc &expr)
{
  // A constructor's member initialiser binds its left side; reading that
  // through would copy the referent instead of pointing at it. Only the right
  // side is a use. clang_cpp_adjust::adjust_side_effect_assign's `#member_init`
  // branch says the same (scope-clang-cpp-irep2.md §3.16).
  if (is_sideeffect_assign2t(expr) && to_sideeffect_assign2t(expr).member_init)
  {
    const sideeffect_assign2t &a = to_sideeffect_assign2t(expr);
    expr2tc rhs = a.rhs;
    if (is_nil_expr(rhs))
      return;

    // Dereferencing the rhs here is only half the story: because the lhs is
    // itself reference-typed, adjust_plain_assignment's c_implicit_typecast
    // then re-wraps this in an address_of via c_typecastt::convert_reference --
    // a same-named function in util/lang/c_typecast.cpp. The round trip is a
    // no-op (`this->__m = &(*m)`), but it spans two translation units.
    convert_reference(ns, rhs);
    if (rhs != a.rhs)
      expr = sideeffect_assign2tc(
        a.type, a.op, a.lhs, rhs, a.location, a.member_init);
    return;
  }

  expr->Foreach_operand([this](expr2tc &op) {
    if (!is_nil_expr(op))
      convert_reference(ns, op);
  });
}

void clang_cpp_adjust_irep2::gen_symbol_code(symbolt &symbol)
{
  // The legacy pass generates these *after* adjusting the body; here they are
  // generated before, so the assignments go through the arms like any other
  // statement rather than being migrated back out and in again.
  gen_vptr_initializations(context, symbol);
}
