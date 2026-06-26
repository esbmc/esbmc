#pragma once

#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <vector>

// Shared IREP2 expression-construction helpers for the Python frontend (V.3).
//
// Each helper performs the legacy<->IREP2 round-trip
//   migrate_expr -> build IREP2 node -> migrate_expr_back
// so the frontend emits nodes pre-lowered for the legacy adjust/goto-convert
// seam. They were originally copy-pasted into ~15 translation units; this
// module is the single source of truth.
//
// Two migrate_type round-trip hazards are guarded uniformly:
//   * a dynamically-sized array type (non-constant size) throws get_width
//     downstream, so the relevant helpers fall back to the legacy constructor;
//   * type attributes such as #cpp_type are dropped by migrate_type, so the
//     member/index/typecast/dereference helpers restore the exact result type
//     (result.type() = t) -- load-bearing e.g. to keep a 1-char string element
//     distinct from an 8-bit int.
namespace python_expr
{
// True iff `t` is, or transitively points to/contains, an array whose size is
// nil or non-constant (a dyn-sized array that does not survive migrate_type).
bool contains_dyn_array(const typet &t);

// Symbol reference `sym`.
exprt build_symbol(const symbolt &sym);

// Typecast `from` to type `t`.
exprt build_typecast(const exprt &from, const typet &t);

// Address-of an lvalue `obj` (symbol/member/index source).
exprt build_address_of(const exprt &obj);

// Dereference `ptr` to a value of type `t`.
exprt build_dereference(const exprt &ptr, const typet &t);

// Struct/union member access `base.name : t`. Falls back to the legacy node
// when the source is not a struct/union/symbol value.
exprt build_member(const exprt &base, const irep_idt &name, const typet &t);

// `(*obj).field : field_type`, where `obj` is a pointer to a struct.
exprt build_deref_member(
  const exprt &obj,
  const irep_idt &field,
  const typet &field_type);

// Array index `arr[idx] : t`. Falls back to the legacy node when the source is
// not an array/vector/symbol value.
exprt build_index(const exprt &arr, const exprt &idx, const typet &t);

// `arr[idx]` with element type taken from the source array's subtype.
exprt build_index(const exprt &arr, const exprt &idx);

// Boolean negation `not op`, `op` a bool-typed value. migrate lowers a legacy
// "not" node to not2tc(migrate(op)), so this is the byte-identical round-trip.
exprt build_not(const exprt &op);

// `a < b` over same-width operands (lessthan2t asserts width consistency).
exprt build_less_than(const exprt &a, const exprt &b);

// `a <= b` over same-width operands (lessthanequal2t asserts width consistency).
exprt build_less_equal(const exprt &a, const exprt &b);

// `a > b` over same-width operands (greaterthan2t asserts width consistency).
exprt build_greater_than(const exprt &a, const exprt &b);

// `a >= b` over same-width operands (greaterthanequal2t asserts width
// consistency).
exprt build_greater_equal(const exprt &a, const exprt &b);

// Boolean disjunction `a || b`, both operands bool-typed. migrate lowers a
// legacy binary "or" node to or2tc(migrate(a), migrate(b)), so this is the
// byte-identical round-trip.
exprt build_or(const exprt &a, const exprt &b);

// `a + b : t` over same-width operands (add2t asserts width consistency).
exprt build_add(const exprt &a, const exprt &b, const typet &t);

// `a - b : t` over same-width operands (sub2t asserts width consistency).
exprt build_sub(const exprt &a, const exprt &b, const typet &t);

// `a * b : t` over same-width operands (mul2t asserts width consistency).
exprt build_mul(const exprt &a, const exprt &b, const typet &t);

// Float `a + b`/`a - b`/`a * b : t` over same-floatbv operands, using the
// default __ESBMC_rounding_mode (matching migrate of a legacy ieee_* node with
// no rounding_mode field). ieee_*2t assert operand-width consistency.
exprt build_ieee_add(const exprt &a, const exprt &b, const typet &t);
exprt build_ieee_sub(const exprt &a, const exprt &b, const typet &t);
exprt build_ieee_mul(const exprt &a, const exprt &b, const typet &t);

// Inequality `a != b` over same-typed operands. migrate lowers a legacy
// "notequal" node to notequal2tc(migrate(a), migrate(b)), so this is the
// byte-identical round-trip.
exprt build_notequal(const exprt &a, const exprt &b);

// Equality `a == b` over same-typed operands. migrate lowers a legacy "="
// node to equality2tc(migrate(a), migrate(b)), so this is the byte-identical
// round-trip.
exprt build_equal(const exprt &a, const exprt &b);

// Expression-context call `fn(args...) : return_type`, fn a function symbol.
exprt build_call_expr(
  const symbolt &fn,
  const typet &return_type,
  const std::vector<exprt> &args);

// Expression-context call `fn_id(args...) : return_type`, the callee referenced
// by name with a placeholder code type.
exprt build_call_expr(
  const irep_idt &fn_id,
  const typet &return_type,
  const std::vector<exprt> &args);

// Expression-context call `callee(args...) : return_type`, callee an already
// built function expression (e.g. a by-name symbol carrying a code_typet).
exprt build_call(
  const exprt &callee,
  const typet &return_type,
  const std::vector<exprt> &args);
} // namespace python_expr
