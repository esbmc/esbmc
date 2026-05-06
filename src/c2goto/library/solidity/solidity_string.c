/* Solidity string operations and conversions */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "solidity_types.h"

char *string_concat(char *x, char *y)
{
__ESBMC_HIDE:;
  size_t xlen = strnlen(x, 256);
  if (xlen < 256)
    snprintf(x + xlen, 256 - xlen, "%s", y);
  return x;
}

char get_char(int digit)
{
__ESBMC_HIDE:;
  char charstr[] = "0123456789ABCDEF";
  return charstr[digit];
}

void sol_rev(char *p)
{
__ESBMC_HIDE:;
  size_t n = strnlen(p, 256);
  if (n == 0)
    return;
  char *q = &p[n - 1];
  char *r = p;
  for (; q > r; q--, r++)
  {
    char s = *q;
    *q = *r;
    *r = s;
  }
}

char *i256toa(int256_t value)
{
__ESBMC_HIDE:;
  // we might have memory leak as we will not free this afterwards
  char *str = (char *)malloc(256 * sizeof(char));
  int256_t base = (int256_t)10;
  unsigned short count = 0;
  bool flag = true;

  if (value < (int256_t)0 && base == (int256_t)10)
  {
    flag = false;
  }
  if (value == (int256_t)0)
  {
    str[count] = '\0';
    return str;
  }
  while (value != (int256_t)0)
  {
    int256_t dig = value % base;
    value -= dig;
    value /= base;

    if (flag == true)
      str[count] = get_char(dig);
    else
      str[count] = get_char(-dig);
    count++;
  }
  if (flag == false)
  {
    str[count] = '-';
    count++;
  }
  str[count] = 0;
  sol_rev(str);
  return str;
}

char *u256toa(uint256_t value)
{
__ESBMC_HIDE:;
  char *str = (char *)malloc(256 * sizeof(char));
  uint256_t base = (uint256_t)10;
  unsigned short count = 0;
  if (value == (uint256_t)0)
  {
    str[count] = '\0';
    return str;
  }
  while (value != (uint256_t)0)
  {
    uint256_t dig = value % base;
    value -= dig;
    value /= base;
    str[count] = get_char(dig);
    count++;
  }
  str[count] = 0;
  sol_rev(str);
  return str;
}

char *decToHexa(int n)
{
__ESBMC_HIDE:;
  char *hexaDeciNum = (char *)malloc(256 * sizeof(char));
  hexaDeciNum[0] = '\0';
  int i = 0;
  while (n != 0)
  {
    int temp = 0;
    temp = n % 16;
    if (temp < 10)
    {
      hexaDeciNum[i] = temp + 48;
      i++;
    }
    else
    {
      hexaDeciNum[i] = temp + 55;
      i++;
    }

    n /= 16;
  }
  char *ans = (char *)malloc(256 * sizeof(char));
  ans[0] = '\0';
  int pos = 0;
  for (int j = i - 1; j >= 0; j--)
  {
    ans[pos] = (char)hexaDeciNum[j];
    pos++;
  }
  ans[pos] = '\0';
  return ans;
}

char *ASCIItoHEX(const char *ascii)
{
__ESBMC_HIDE:;
  char *hex = (char *)malloc(256 * sizeof(char));
  hex[0] = '\0';
  size_t n = strnlen(ascii, 256);
  for (size_t i = 0; i < n; i++)
  {
    char ch = ascii[i];
    int tmp = (int)ch;
    char *part = decToHexa(tmp);
    strcat(hex, part);
  }
  return hex;
}

uint256_t hexdec(const char *hex)
{
__ESBMC_HIDE:;
  /*https://stackoverflow.com/questions/10324/convert-a-hexadecimal-string-to-an-integer-efficiently-in-c*/

  static const long hextable[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,
    9,  -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1};
  uint256_t ret = 0;
  while (*hex && ret >= (uint256_t)0)
  {
    ret = (ret << (uint256_t)4) | (uint256_t)hextable[*hex++];
  }
  return ret;
}

uint256_t str2uint(const char *str)
{
__ESBMC_HIDE:;
  // Pure: same string contents -> same uint256_t.
  // The previous hexdec(ASCIItoHEX(str)) implementation went through malloc,
  // so symex could not treat repeated calls on the same literal as equal —
  // mapping(string => T) writes and reads produced different keys, missing
  // the just-stored entry.
  //
  // Encoding: pack the first 32 bytes of str into a 256-bit big-endian
  // integer, stopping at the NUL terminator. Strings longer than 32 bytes
  // are still mapped deterministically (further bytes shift earlier ones
  // out of the high 32 lanes), preserving collision resistance.
  uint256_t result = (uint256_t)0;
  for (size_t i = 0; str[i] != '\0' && i < 32; i++)
    result = (result << (uint256_t)8) | (uint256_t)(uint8_t)str[i];
  return result;
}

// Fold a 256-bit mapping key down to a 64-bit array-domain index.
// Implemented as a single C function so the frontend only generates one call,
// avoiding side-effect duplication when the key is a function-call result
// (e.g. str2uint, bytes_static_to_mapping_key) that would otherwise be
// re-evaluated per chunk.
uint64_t _ESBMC_str_key_fold64(uint256_t key)
{
__ESBMC_HIDE:;
  uint64_t k0 = (uint64_t)key;
  uint64_t k1 = (uint64_t)(key >> (uint256_t)64);
  uint64_t k2 = (uint64_t)(key >> (uint256_t)128);
  uint64_t k3 = (uint64_t)(key >> (uint256_t)192);
  return k0 ^ k1 ^ k2 ^ k3;
}

// string assign
void _str_assign(char **str1, const char *str2)
{
__ESBMC_HIDE:;
  // Ensure str1 is a valid pointer (not NULL)
  if (str1 == NULL)
  {
    return; // Early exit if str1 is invalid
  }
  // Free *str1 only if it was previously allocated (non-NULL)
  // if (*str1 != NULL) {
  //     free(*str1);
  // }

  // If str2 is NULL, set *str1 to NULL (avoid dangling pointers)
  if (str2 == NULL)
  {
    *str1 = NULL;
    return;
  }
  size_t len = strnlen(str2, SIZE_MAX - 1);
  *str1 = (char *)malloc(len + 1);
  memcpy(*str1, str2, len);
  (*str1)[len] = '\0';
}

unsigned int nondet_uint();
char nondet_char();

__attribute__((annotate("__ESBMC_inf_size"))) char _ESBMC_rand_str[1];

char *nondet_string()
{
__ESBMC_HIDE:;
  size_t len = nondet_uint();
  // __ESBMC_assume(len < SIZE_MAX);

  for (size_t i = 0; i < len; ++i)
  {
    _ESBMC_rand_str[i] = nondet_char();
    __ESBMC_assume(_ESBMC_rand_str[i] != '\0');
  }
  _ESBMC_rand_str[len] = '\0';
  return _ESBMC_rand_str;
}
