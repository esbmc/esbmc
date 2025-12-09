#ifndef CPROVER_C_TYPES_H
#define CPROVER_C_TYPES_H

#include <util/expr.h>
#include <irep2/irep2.h>

typet index_type();
type2tc index_type2();

typet int_type();
type2tc int_type2();
typet uint_type();
type2tc uint_type2();
typet int128_type();
type2tc int128_type2();
typet uint128_type();
type2tc uint128_type2();
typet uint256_type();
typet long_int_type();
type2tc long_int_type2();

typet long_long_int_type();
type2tc long_long_int_type2();

typet long_uint_type();
type2tc long_uint_type2();

typet long_long_uint_type();
type2tc long_long_uint_type2();

typet wchar_type();
type2tc char_type2();

typet float_type();
type2tc float_type2();

typet build_float_type(unsigned width);
typet double_type();
type2tc double_type2();

typet long_double_type();
type2tc long_double_type2();

typet pointer_type();
type2tc pointer_type2();

type2tc ptraddr_type2();

type2tc bitsize_type2();

typet enum_type();
typet signed_short_int_type();
typet unsigned_short_int_type();
typet char_type();
typet unsigned_char_type();
typet signed_char_type();
typet char16_type();
typet char32_type();
typet unsigned_wchar_type();
typet half_float_type();
typet size_type();
typet signed_size_type();
typet bool_type();

type2tc get_uint8_type();
type2tc get_uint16_type();
type2tc get_uint32_type();
type2tc get_uint64_type();
type2tc get_int8_type();
type2tc get_int16_type();
type2tc get_int32_type();
type2tc get_int64_type();
type2tc get_uint_type(unsigned int sz);
type2tc get_int_type(unsigned int sz);
type2tc get_bool_type();
type2tc get_empty_type();

type2tc size_type2();
type2tc signed_size_type2();

#endif
