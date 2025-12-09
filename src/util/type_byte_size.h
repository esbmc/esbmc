#ifndef CPROVER_POINTER_OFFSET_SIZE_H
#define CPROVER_POINTER_OFFSET_SIZE_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/mp_arith.h>
#include <util/namespace.h>
#include <util/std_types.h>

BigInt member_offset_bits(
  const type2tc &type,
  const irep_idt &member,
  const namespacet *ns = nullptr);
BigInt member_offset(
  const type2tc &type,
  const irep_idt &member,
  const namespacet *ns = nullptr);

/* These can throw array_type2t::inf_sized_array_excp or
 * array_type2t::dyn_sized_array_excp */
BigInt type_byte_size_bits(const type2tc &type, const namespacet *ns = nullptr);
BigInt type_byte_size(const type2tc &type, const namespacet *ns = nullptr);
BigInt type_byte_size_default(
  const type2tc &type,
  const BigInt &defaultval,
  const namespacet *ns = nullptr);

/* type_byte_size*_expr() can throw array_type2t::inf_sized_array_excp */
expr2tc
type_byte_size_bits_expr(const type2tc &type, const namespacet *ns = nullptr);
expr2tc
type_byte_size_expr(const type2tc &type, const namespacet *ns = nullptr);

expr2tc
compute_pointer_offset(const expr2tc &expr, const namespacet *ns = nullptr);
expr2tc compute_pointer_offset_bits(
  const expr2tc &expr,
  const namespacet *ns = nullptr);

const expr2tc &get_base_object(const expr2tc &expr);

#endif
