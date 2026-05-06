#ifndef SOLIDITY_TYPES_H
#define SOLIDITY_TYPES_H

#include <stdbool.h>
#include <stddef.h>

#if defined(__clang__)
#  if __clang_major__ >= 16
#    define BIGINT(bits) _BitInt(bits)
#  elif __clang_major__ >= 11 && __clang_major__ <= 13
#    define BIGINT(bits) _ExtInt(bits)
#  else
#    error "Unsupported Clang version: _ExtInt/_BitInt not available."
#  endif
#else
#  error "This code requires Clang to compile."
#endif

typedef BIGINT(256) int256_t;
typedef unsigned BIGINT(256) uint256_t;
typedef unsigned BIGINT(160) address_t;

/* Dynamic bytes type — shared between solidity_bytes.c and solidity_builtins.c */
typedef struct BytesDynamic
{
  size_t offset;
  size_t length;
  size_t capacity;
  int initialized;
} BytesDynamic;

struct sol_llc_ret
{
  unsigned int x;
  unsigned int y;
};

#endif /* SOLIDITY_TYPES_H */
