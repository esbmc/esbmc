#ifndef CPROVER_MP_ARITH_H
#define CPROVER_MP_ARITH_H

#include <big-int/bigint.hh>
#include <ostream>
#include <string>

std::ostream &operator<<(std::ostream &, const BigInt &);
BigInt operator>>(const BigInt &, const BigInt &);
BigInt operator<<(const BigInt &, const BigInt &);

const std::string integer2string(const BigInt &, unsigned base = 10);
const BigInt string2integer(const std::string &, unsigned base = 10);
const std::string integer2binary(const BigInt &, std::size_t width);
const BigInt binary2integer(const std::string &, bool is_signed);

#endif // CPROVER_UTIL_MP_ARITH_H
