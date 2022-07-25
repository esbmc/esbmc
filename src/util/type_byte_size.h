#ifndef CPROVER_POINTER_OFFSET_SIZE_H
#define CPROVER_POINTER_OFFSET_SIZE_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/mp_arith.h>
#include <util/namespace.h>
#include <util/std_types.h>

BigInt member_offset_bits(const type2tc &type, const irep_idt &member);
BigInt member_offset(const type2tc &type, const irep_idt &member);

BigInt type_byte_size_bits(const type2tc &type);
BigInt type_byte_size(const type2tc &type);
BigInt type_byte_size_default(const type2tc &type, const BigInt &defaultval);

expr2tc compute_pointer_offset(const expr2tc &expr);
expr2tc compute_pointer_offset_bits(const expr2tc &expr);

const expr2tc &get_base_object(const expr2tc &expr);
const irep_idt get_string_argument(const expr2tc &expr);

#endif
