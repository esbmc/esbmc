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
  strncat(x, y, 256);
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
  char *q = &p[strlen(p) - 1];
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
  for (int i = 0; i < strlen(ascii); i++)
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
  return hexdec(ASCIItoHEX(str));
}

#define _ESBMC_SOL_STR_MAX 64

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
  /* Bounded scan instead of strlen: in --contract mode (and anywhere a
   * string parameter is nondet) str2's bytes are symbolic, so the generic
   * strlen's "while (s[len] != 0) len++" has no concrete termination and
   * the unwinder spins until OOM. A 256-byte cap is more than enough for
   * any Solidity name/symbol/short-URI literal we see in practice. */
  size_t len = 0;
  while (len < _ESBMC_SOL_STR_MAX && str2[len] != '\0')
    ++len;
  *str1 = (char *)malloc(len + 1);
  for (size_t i = 0; i < _ESBMC_SOL_STR_MAX; ++i)
  {
    if (i >= len)
      break;
    (*str1)[i] = str2[i];
  }
  (*str1)[len] = '\0';
}

unsigned int nondet_uint();
char nondet_char();

/* Fixed-size buffer (not __ESBMC_inf_size) so a concrete NUL at index
 * _ESBMC_NONDET_STRING_MAX guarantees strlen() terminates without the
 * generic scan loop unwinding indefinitely. */
#define _ESBMC_NONDET_STRING_MAX 32
char _ESBMC_rand_str[_ESBMC_NONDET_STRING_MAX + 1];

/* Cap the symbolic length so symex terminates without --unwind. Without
 * a concrete loop bound, len is symbolic and the fill loop would unwind
 * forever; without a zero-initialised tail, strlen() on the returned
 * string would keep scanning symbolic bytes past the NUL until OOM. */
char *nondet_string()
{
__ESBMC_HIDE:;
  size_t len = nondet_uint();
  __ESBMC_assume(len < _ESBMC_NONDET_STRING_MAX);

  /* Zero the whole buffer concretely so any later strlen() terminates at
   * or before _ESBMC_NONDET_STRING_MAX.  memset here is a constant size
   * on a writable global with simplify on => intrinsic_memset single-
   * shot (no per-byte loop), so the k-induction base case / a sub-33
   * --unwind cannot spuriously prune the post-string path by k-bounding
   * this fixed-trip-count library loop (was: 33-iter zero-fill loop;
   * see notes/Results/branch_cov/STAGE5_RESIDUAL_DIAG.md Stage H). */
  memset(_ESBMC_rand_str, 0, _ESBMC_NONDET_STRING_MAX + 1);

  /* Concrete upper bound in the loop header gives the unwinder a static
   * stop; the extra i<len lets the loop exit early for smaller lengths. */
  for (size_t i = 0; i < _ESBMC_NONDET_STRING_MAX; ++i)
  {
    if (i >= len)
      break;
    _ESBMC_rand_str[i] = nondet_char();
    __ESBMC_assume(_ESBMC_rand_str[i] != '\0');
  }
  return _ESBMC_rand_str;
}
