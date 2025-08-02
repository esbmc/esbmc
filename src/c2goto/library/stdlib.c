#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <stdint.h> /* uintptr_t */
#include <math.h>
#include <stdbool.h>

#include <assert.h>

#undef errno
extern _Thread_local int errno;

#undef exit
#undef abort
#undef calloc
#undef getenv
#undef atoi
#undef atol
#undef atoll

typedef struct atexit_key
{
  void (*atexit_func)();
} __ESBMC_atexit_key;

// Infinite array for atexit functions
__attribute__((annotate(
  "__ESBMC_inf_size"))) static __ESBMC_atexit_key __ESBMC_stdlib_atexit_key[1];
static size_t __ESBMC_atexit_count = 0;

void __ESBMC_atexit_handler()
{
__ESBMC_HIDE:;
  while (__ESBMC_atexit_count > 0)
  {
    __ESBMC_atexit_count--;
    __ESBMC_stdlib_atexit_key[__ESBMC_atexit_count].atexit_func();
  }
}

int atexit(void (*func)(void))
{
__ESBMC_HIDE:;
  __ESBMC_stdlib_atexit_key[__ESBMC_atexit_count].atexit_func = func;
  __ESBMC_atexit_count++;
  return 0;
}

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-noreturn"
void exit(int status)
{
__ESBMC_HIDE:;
  __ESBMC_atexit_handler();
  __ESBMC_memory_leak_checks();
  __ESBMC_assume(0);
}

_Bool __ESBMC_no_abnormal_memory_leak(void);

void abort(void)
{
__ESBMC_HIDE:;
  if (!__ESBMC_no_abnormal_memory_leak())
    __ESBMC_memory_leak_checks();
  __ESBMC_assume(0);
}

void _Exit(int status)
{
__ESBMC_HIDE:;
  __ESBMC_memory_leak_checks();
  __ESBMC_assume(0);
}
#pragma clang diagnostic pop

void *calloc(size_t nmemb, size_t size)
{
__ESBMC_HIDE:;
  if (!nmemb)
    return NULL;

  size_t total_size = nmemb * size;
  void *res = malloc(total_size);
  if (res)
    memset(res, 0, total_size);
  return res;
}

long int strtol(const char *str, char **endptr, int base)
{
__ESBMC_HIDE:;
  long int result = 0;
  int sign = 1;

  // Handle whitespace
  while (isspace(*str))
    str++;

  // Handle sign
  if (*str == '-')
  {
    sign = -1;
    str++;
  }
  else if (*str == '+')
    str++;

  // Handle base
  if (base == 0)
  {
    if (*str == '0')
    {
      base = 8;
      if (tolower(str[1]) == 'x')
      {
        base = 16;
        str += 2;
      }
      else
        str++;
    }
    else
      base = 10;
  }
  else if (base == 16 && *str == '0' && tolower(str[1]) == 'x')
    str += 2;

  // Convert digits
  while (isdigit(*str) || (base == 16 && isxdigit(*str)))
  {
    int digit = tolower(*str) - '0';
    if (digit > 9)
      digit -= 7;
    if (result > (LONG_MAX - digit) / base)
      return sign == -1 ? LONG_MIN : LONG_MAX;
    result = result * base + digit;
    str++;
  }

  // Set end pointer
  if (endptr != NULL)
    *endptr = (char *)str;

  return sign * result;
}

float strtof(const char *str, char **endptr)
{
__ESBMC_HIDE:;
  float result = 0.0f;
  int sign = 1;
  float decimal_factor = 0.1f;

  while (isspace(*str))
    str++;

  if (*str == '-')
  {
    sign = -1;
    str++;
  }
  else if (*str == '+')
  {
    str++;
  }

  while (isdigit(*str))
  {
    result = result * 10 + (*str - '0');
    str++;
  }

  if (*str == '.')
  {
    str++;
    while (isdigit(*str))
    {
      result += (*str - '0') * decimal_factor;
      decimal_factor /= 10;
      str++;
    }
  }

  if (*str == 'e' || *str == 'E')
  {
    str++;
    int exp_sign = 1;
    if (*str == '-')
    {
      exp_sign = -1;
      str++;
    }
    else if (*str == '+')
    {
      str++;
    }

    int exponent = 0;
    while (isdigit(*str))
    {
      exponent = exponent * 10 + (*str - '0');
      str++;
    }

    result *= pow(10, exp_sign * exponent);
  }

  result *= sign;

  if (endptr != NULL)
    *endptr = (char *)str;

  return result;
}

/* one plus the numeric value, rest is zero */
static const unsigned char get_atoi_map(unsigned char pos)
{
__ESBMC_HIDE:;
  const unsigned char ATOI_MAP[256] = {
    ['0'] = 1,
    ['1'] = 2,
    ['2'] = 3,
    ['3'] = 4,
    ['4'] = 5,
    ['5'] = 6,
    ['6'] = 7,
    ['7'] = 8,
    ['8'] = 9,
    ['9'] = 10,
  };
  return ATOI_MAP[pos];
}

#define ATOI_DEF(name, type, TYPE)                                             \
  type name(const char *s)                                                     \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    while (isspace(*s))                                                        \
      s++;                                                                     \
    int neg = 0;                                                               \
    if (*s == '-')                                                             \
    {                                                                          \
      neg = 1;                                                                 \
      s++;                                                                     \
    }                                                                          \
    else if (*s == '+')                                                        \
      s++;                                                                     \
    unsigned type r = 0;                                                       \
    for (unsigned char c; (c = get_atoi_map((unsigned char)*s)); s++)          \
    {                                                                          \
      c--;                                                                     \
      if (r > (TYPE##_MAX - c) / 10)                                           \
        return neg ? TYPE##_MIN : TYPE##_MAX;                                  \
      r *= 10;                                                                 \
      r += c;                                                                  \
    }                                                                          \
    return neg ? -r : r;                                                       \
  }

ATOI_DEF(atoi, int, INT)
ATOI_DEF(atol, long, LONG)
ATOI_DEF(atoll, long long, LLONG)

#undef ATOI_DEF

char *getenv(const char *name)
{
__ESBMC_HIDE:;

  __ESBMC_assert(name != NULL, "getenv called with NULL pointer");

  // Return NULL when called with an empty string parameter
  if (*name == '\0')
    return NULL;

  // Return NULL when the environment variable name
  // contains an equals sign (=), per POSIX specification
  if (strchr(name, '=') != NULL)
    return NULL;

  // Non-deterministically model whether the variable exists
  _Bool found = __ESBMC_nondet_bool();
  if (!found)
    return NULL;

  char *buffer;
  size_t buf_size;

  __ESBMC_assume(buf_size >= 1);
  buffer = (char *)__ESBMC_alloca(buf_size);
  buffer[buf_size - 1] = 0;
  return buffer;
}

void *ldv_malloc(size_t size)
{
__ESBMC_HIDE:;
  return malloc(size);
}

void *ldv_zalloc(size_t size)
{
__ESBMC_HIDE:;
  return malloc(size);
}

size_t strlcat(char *dst, const char *src, size_t siz)
{
__ESBMC_HIDE:;
  char *d = dst;
  const char *s = src;
  size_t n = siz;
  size_t dlen;

  /* Find the end of dst and adjust bytes left but don't go past end */
  while (n-- != 0 && *d != '\0')
    d++;
  dlen = d - dst;
  n = siz - dlen;

  if (n == 0)
    return (dlen + strlen(s));
  while (*s != '\0')
  {
    if (n != 1)
    {
      *d++ = *s;
      n--;
    }
    s++;
  }
  *d = '\0';

  return (dlen + (s - src)); /* count does not include NUL */
}

int posix_memalign(void **memptr, size_t align, size_t size)
{
__ESBMC_HIDE:;
  if (
    !align || (align & (align - 1)) || /* alignment must be a power of 2 */
    (size & (align - 1)) /* size must be a multiple of alignment */
  )
    return EINVAL;
  int save = errno;
  void *r = malloc(size);
  errno = save;
  __ESBMC_assume(!((uintptr_t)r & (align - 1)));
  if (size && !r)
    return ENOMEM;
  *memptr = r;
  return 0;
}

void *aligned_alloc(size_t align, size_t size)
{
__ESBMC_HIDE:;
  void *r = NULL;
  errno = posix_memalign(&r, align, size);
  return r;
}

int rand(void)
{
__ESBMC_HIDE:;
  return nondet_uint() % ((unsigned)RAND_MAX + 1);
}

long random(void)
{
__ESBMC_HIDE:;
  return nondet_ulong() % ((unsigned)INT32_MAX + 1);
}

#if 0
void srand (unsigned int s)
{
	seed = s;
}
#endif

void rev(char *p)
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

char *itoa(int value, char *str, int base)
{
__ESBMC_HIDE:;
  int count = 0;
  bool flag = true;
  if (value < 0 && base == 10)
  {
    flag = false;
    value = -value;
  }

  if (value == 0)
    str[count++] = '0';

  while (value != 0)
  {
    int dig = value % base;
    if (dig < 10)
      str[count++] = '0' + dig;
    else
      str[count++] = 'a' + (dig - 10);

    value /= base;
  }

  if (!flag)
    str[count++] = '-';

  str[count] = '\0';

  rev(str);

  return str;
}
