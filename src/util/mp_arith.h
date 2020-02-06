/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_MP_ARITH_H
#define CPROVER_MP_ARITH_H

#include <big-int/bigint.hh>
#include <iostream>
#include <string>

std::ostream &operator<<(std::ostream &out, const BigInt &n);
BigInt operator>>(const BigInt &a, const BigInt &b);
BigInt operator<<(const BigInt &a, const BigInt &b);

BigInt bitwise_or(const BigInt &a, const BigInt &b);
BigInt bitwise_and(const BigInt &a, const BigInt &b);
BigInt bitwise_xor(const BigInt &a, const BigInt &b);

BigInt
arith_left_shift(const BigInt &a, const BigInt &b, std::size_t true_size);
BigInt
arith_right_shift(const BigInt &a, const BigInt &b, std::size_t true_size);
BigInt
logic_left_shift(const BigInt &a, const BigInt &b, std::size_t true_size);
BigInt
logic_right_shift(const BigInt &a, const BigInt &b, std::size_t true_size);
BigInt rotate_right(const BigInt &a, const BigInt &b, std::size_t true_size);
BigInt rotate_left(const BigInt &a, const BigInt &b, std::size_t true_size);

const std::string integer2string(const BigInt &n, unsigned base = 10);
const BigInt string2integer(const std::string &n, unsigned base = 10);
const std::string integer2binary(const BigInt &n, std::size_t width);
const BigInt binary2integer(const std::string &n, bool is_signed);
std::size_t integer2size_t(const BigInt &);
unsigned integer2unsigned(const BigInt &);

#endif
