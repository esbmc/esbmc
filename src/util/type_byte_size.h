/*******************************************************************\

Module: Pointer Logic

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_OFFSET_SIZE_H
#define CPROVER_POINTER_OFFSET_SIZE_H

#include <mp_arith.h>
#include <expr.h>
#include <namespace.h>
#include <std_types.h>
#include <irep2.h>

mp_integer member_offset(const type2tc &type, const irep_idt &member);

mp_integer type_byte_size(const type2tc &type);
mp_integer type_byte_size_default(const type2tc &type, mp_integer defaultval);

expr2tc compute_pointer_offset(const expr2tc &expr);

const expr2tc & get_base_object(const expr2tc &expr);

#endif
