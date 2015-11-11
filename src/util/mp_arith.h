/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_MP_ARITH_H
#define CPROVER_MP_ARITH_H

#include <string>
#include <iostream>

#include "big-int/bigint.hh"

typedef BigInt mp_integer;
typedef unsigned int u_int;

std::ostream& operator<<(std::ostream& out, const mp_integer &n);
mp_integer operator>>(const mp_integer &a, unsigned int b);
mp_integer operator<<(const mp_integer &a, unsigned int b);

const std::string integer2string(const mp_integer &n, unsigned base=10);
const mp_integer string2integer(const std::string &n, unsigned base=10);
const std::string integer2binary(const mp_integer &n, unsigned width);
const mp_integer binary2integer(const std::string &n, bool is_signed);
const mp_integer extract_fraction(const std::string &n, bool is_signed, u_int from, u_int to);
unsigned long integer2long(const mp_integer &n);

#endif
