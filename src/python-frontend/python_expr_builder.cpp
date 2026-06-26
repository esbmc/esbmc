#include <python-frontend/python_expr_builder.h>

#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/std_expr.h>
#include <util/std_types.h>
#include <util/std_code.h>
#include <util/expr_util.h>

namespace python_expr
{
bool contains_dyn_array(const typet &t)
{
  if (t.is_array())
  {
    const array_typet &at = to_array_type(t);
    if (at.size().is_nil() || !at.size().is_constant())
      return true;
    return contains_dyn_array(at.subtype());
  }
  if (t.is_pointer())
    return contains_dyn_array(t.subtype());
  return false;
}

exprt build_symbol(const symbolt &sym)
{
  if (contains_dyn_array(sym.get_type()))
    return symbol_expr(sym);
  return migrate_expr_back(symbol_expr2tc(sym));
}

exprt build_typecast(const exprt &from, const typet &t)
{
  if (contains_dyn_array(t) || contains_dyn_array(from.type()))
    return typecast_exprt(from, t);
  expr2tc from2;
  migrate_expr(from, from2);
  exprt result = migrate_expr_back(typecast2tc(migrate_type(t), from2));
  // migrate_type does not round-trip #cpp_type; restore the exact target type
  // so legacy typecast_exprt(from, t) is reproduced faithfully.
  result.type() = t;
  return result;
}

// address_of2t's sources here are lvalues (symbols/members/indices), so no
// guard is needed beyond dyn-array.
exprt build_address_of(const exprt &obj)
{
  if (contains_dyn_array(obj.type()))
    return address_of_exprt(obj);
  expr2tc obj2;
  migrate_expr(obj, obj2);
  return migrate_expr_back(address_of2tc(obj2->type, obj2));
}

exprt build_dereference(const exprt &ptr, const typet &t)
{
  if (contains_dyn_array(t))
    return dereference_exprt(ptr, t);
  expr2tc ptr2;
  migrate_expr(ptr, ptr2);
  exprt result = migrate_expr_back(dereference2tc(migrate_type(t), ptr2));
  // migrate_type does not round-trip #cpp_type; restore the exact target type
  // so legacy dereference_exprt(t)+op0=ptr is reproduced faithfully.
  result.type() = t;
  return result;
}

// member2t needs a struct/union/symbol source; fall back to the legacy node
// otherwise (and for dyn-array result/base types).
exprt build_member(const exprt &base, const irep_idt &name, const typet &t)
{
  if (contains_dyn_array(t) || contains_dyn_array(base.type()))
    return member_exprt(base, name, t);
  expr2tc base2;
  migrate_expr(base, base2);
  if (
    is_struct_type(base2->type) || is_union_type(base2->type) ||
    is_symbol_type(base2->type))
  {
    exprt result = migrate_expr_back(member2tc(migrate_type(t), base2, name));
    // migrate_type does not round-trip #cpp_type; restore the exact member type.
    result.type() = t;
    return result;
  }
  return member_exprt(base, name, t);
}

// Build (*obj).field : field_type. `obj` is a pointer to a struct, so the
// dereferenced struct is the resolved member source.
exprt build_deref_member(
  const exprt &obj,
  const irep_idt &field,
  const typet &field_type)
{
  expr2tc obj2;
  migrate_expr(obj, obj2);
  expr2tc deref2 = dereference2tc(migrate_type(obj.type().subtype()), obj2);
  return migrate_expr_back(member2tc(migrate_type(field_type), deref2, field));
}

// index2t needs an array/vector/symbol source; fall back to the legacy node
// otherwise (and for dyn-array source/result types -- string indexing relies on
// the #cpp_type attribute that migrate_type drops, hence result.type() = t).
exprt build_index(const exprt &arr, const exprt &idx, const typet &t)
{
  if (contains_dyn_array(arr.type()) || contains_dyn_array(t))
    return index_exprt(arr, idx, t);
  expr2tc arr2, idx2;
  migrate_expr(arr, arr2);
  migrate_expr(idx, idx2);
  if (
    is_array_type(arr2->type) || is_vector_type(arr2->type) ||
    is_symbol_type(arr2->type))
  {
    exprt result = migrate_expr_back(index2tc(migrate_type(t), arr2, idx2));
    result.type() = t;
    return result;
  }
  return index_exprt(arr, idx, t);
}

exprt build_index(const exprt &arr, const exprt &idx)
{
  return build_index(arr, idx, arr.type().subtype());
}

// `not op`, `op` a bool-typed value. migrate lowers a legacy "not" node to
// not2tc(migrate(op)) (util/migrate.cpp i_not path), so this is the
// byte-identical round-trip.
exprt build_not(const exprt &op)
{
  expr2tc op2;
  migrate_expr(op, op2);
  return migrate_expr_back(not2tc(op2));
}

// Shared round-trip for the binary builders below: migrate both legacy operands
// to IREP2, build the node via `make`, and back-migrate. migrate lowers each
// legacy binary node (`<`, `<=`, `>`, `>=`, `or`, `+`, `-`) to the matching
// *2tc over the migrated operands with no coercion, so every builder is the
// byte-identical round-trip of the node goto-convert would produce.
namespace
{
template <typename Make>
exprt migrate_binary(const exprt &a, const exprt &b, Make make)
{
  expr2tc a2, b2;
  migrate_expr(a, a2);
  migrate_expr(b, b2);
  return migrate_expr_back(make(a2, b2));
}

// As migrate_binary, for a typed node make(migrate_type(t), a, b). Also restores
// the exact result type, which migrate_type drops (e.g. #cpp_type).
template <typename Make>
exprt migrate_typed_binary(
  const exprt &a,
  const exprt &b,
  const typet &t,
  Make make)
{
  exprt result = migrate_binary(a, b, [&](const expr2tc &x, const expr2tc &y) {
    return make(migrate_type(t), x, y);
  });
  result.type() = t;
  return result;
}
} // namespace

exprt build_less_than(const exprt &a, const exprt &b)
{
  return migrate_binary(
    a, b, [](const expr2tc &x, const expr2tc &y) { return lessthan2tc(x, y); });
}

exprt build_less_equal(const exprt &a, const exprt &b)
{
  return migrate_binary(a, b, [](const expr2tc &x, const expr2tc &y) {
    return lessthanequal2tc(x, y);
  });
}

exprt build_greater_than(const exprt &a, const exprt &b)
{
  return migrate_binary(a, b, [](const expr2tc &x, const expr2tc &y) {
    return greaterthan2tc(x, y);
  });
}

exprt build_greater_equal(const exprt &a, const exprt &b)
{
  return migrate_binary(a, b, [](const expr2tc &x, const expr2tc &y) {
    return greaterthanequal2tc(x, y);
  });
}

// `a || b`, both operands bool-typed.
exprt build_or(const exprt &a, const exprt &b)
{
  return migrate_binary(
    a, b, [](const expr2tc &x, const expr2tc &y) { return or2tc(x, y); });
}

exprt build_add(const exprt &a, const exprt &b, const typet &t)
{
  return migrate_typed_binary(
    a, b, t, [](const type2tc &ty, const expr2tc &x, const expr2tc &y) {
      return add2tc(ty, x, y);
    });
}

exprt build_sub(const exprt &a, const exprt &b, const typet &t)
{
  return migrate_typed_binary(
    a, b, t, [](const type2tc &ty, const expr2tc &x, const expr2tc &y) {
      return sub2tc(ty, x, y);
    });
}

exprt build_mul(const exprt &a, const exprt &b, const typet &t)
{
  return migrate_typed_binary(
    a, b, t, [](const type2tc &ty, const expr2tc &x, const expr2tc &y) {
      return mul2tc(ty, x, y);
    });
}

// `a != b` over same-typed operands. migrate lowers a legacy "notequal" node to
// notequal2tc(migrate(a), migrate(b)) (util/migrate.cpp notequal path), so this
// is the byte-identical round-trip.
exprt build_notequal(const exprt &a, const exprt &b)
{
  expr2tc a2, b2;
  migrate_expr(a, a2);
  migrate_expr(b, b2);
  return migrate_expr_back(notequal2tc(a2, b2));
}

// Expression-context call `fn(args...)` returning return_type. If the return
// type or any argument type contains a dyn-sized array (which does not
// round-trip), build the legacy side_effect_expr_function_callt instead.
exprt build_call_expr(
  const symbolt &fn,
  const typet &return_type,
  const std::vector<exprt> &args)
{
  bool dyn = contains_dyn_array(return_type);
  for (const exprt &a : args)
    dyn = dyn || contains_dyn_array(a.type());
  if (dyn)
  {
    side_effect_expr_function_callt call;
    call.function() = build_symbol(fn);
    for (const exprt &a : args)
      call.arguments().push_back(a);
    call.type() = return_type;
    return call;
  }
  std::vector<expr2tc> args2;
  args2.reserve(args.size());
  for (const exprt &a : args)
  {
    expr2tc a2;
    migrate_expr(a, a2);
    args2.push_back(std::move(a2));
  }
  return migrate_expr_back(side_effect_function_call2tc(
    migrate_type(return_type), symbol_expr2tc(fn), args2));
}

exprt build_call_expr(
  const irep_idt &fn_id,
  const typet &return_type,
  const std::vector<exprt> &args)
{
  return build_call(symbol_exprt(fn_id, code_typet()), return_type, args);
}

exprt build_call(
  const exprt &callee,
  const typet &return_type,
  const std::vector<exprt> &args)
{
  bool dyn = contains_dyn_array(return_type);
  for (const exprt &a : args)
    dyn = dyn || contains_dyn_array(a.type());
  if (dyn)
  {
    side_effect_expr_function_callt call(return_type);
    call.function() = callee;
    for (const exprt &a : args)
      call.arguments().push_back(a);
    return call;
  }
  expr2tc callee2;
  migrate_expr(callee, callee2);
  std::vector<expr2tc> args2;
  args2.reserve(args.size());
  for (const exprt &a : args)
  {
    expr2tc a2;
    migrate_expr(a, a2);
    args2.push_back(std::move(a2));
  }
  return migrate_expr_back(
    side_effect_function_call2tc(migrate_type(return_type), callee2, args2));
}
} // namespace python_expr
