#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>

#undef exit
#undef abort
#undef calloc
#undef getenv

typedef struct atexit_key
{
  void (*atexit_func)();
  struct atexit_key *next;
} __ESBMC_atexit_key;

static __ESBMC_atexit_key *__ESBMC_atexits = NULL;

void __atexit_handler()
{
__ESBMC_HIDE:;
  // This is here to prevent k-induction from unwind the next loop unnecessarily
  if(__ESBMC_atexits == NULL)
    return;

  while(__ESBMC_atexits)
  {
    __ESBMC_atexits->atexit_func();
    __ESBMC_atexit_key *__ESBMC_tmp = __ESBMC_atexits->next;
    free(__ESBMC_atexits);
    __ESBMC_atexits = __ESBMC_tmp;
  }
}

int atexit(void (*func)(void))
{
__ESBMC_HIDE:;
  __ESBMC_atexit_key *l =
    (__ESBMC_atexit_key *)malloc(sizeof(__ESBMC_atexit_key));
  if(l == NULL)
    return -1;
  l->atexit_func = func;
  l->next = __ESBMC_atexits;
  __ESBMC_atexits = l;
  return 0;
}

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-noreturn"
void exit(int status)
{
__ESBMC_HIDE:;
  __atexit_handler();
  __ESBMC_memory_leak_checks();
  __ESBMC_assume(0);
}

void abort(void)
{
__ESBMC_HIDE:;
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
  if(!nmemb)
    return NULL;

  size_t total_size = nmemb * size;
  void *res = malloc(total_size);
  if(res)
    memset(res, 0, total_size);
  return res;
}

#if 0
long strtol(const char *nptr, char **endptr, int base)
{
__ESBMC_HIDE:;
  if(base == 1 || base < 0 || base > 36)
    return 0;

  long res = 0;
  _Bool in_number = 0;
  char sign = 0;

  // 32 chars is an arbitrarily chosen limit
  int i = 0;
  for(; i < 31; ++i)
  {
    char ch = nptr[i];
    char sub = 0;
    if(ch == 0)
      break;
    else if(
      (base == 0 || base == 16) && !in_number && ch == '0' &&
      (nptr[i + 1] == 'x' || nptr[i + 1] == 'X'))
    {
      base = 16;
      in_number = 1;
      ++i;
      continue;
    }
    else if(base == 0 && !in_number && ch == '0')
    {
      base = 8;
      in_number = 1;
      continue;
    }
    else if(!in_number && !sign && isspace(ch))
      continue;
    else if(!in_number && !sign && (ch == '-' || ch == '+'))
    {
      sign = ch;
      continue;
    }
    else if(base > 10 && ch >= 'a' && ch - 'a' < base - 10)
      sub = 'a' - 10;
    else if(base > 10 && ch >= 'A' && ch - 'A' < base - 10)
      sub = 'A' - 10;
    else if(isdigit(ch))
    {
      sub = '0';
      base = base == 0 ? 10 : base;
    }
    else
      break;

    in_number = 1;
    _Bool overflow = __ESBMC_overflow_smull(res, (long)base, &res);
    if(overflow || __ESBMC_overflow_saddl(res, (long)(ch - sub), &res))
    {
      if(sign == '-')
        return LONG_MIN;
      else
        return LONG_MAX;
    }
  }

  if(endptr != 0)
    *endptr = (char *)nptr + i;

  if(sign == '-')
    res *= -1;

  return res;
}
#endif
#if 1
long int strtol(const char *str, char **endptr, int base)
{
  long int result = 0;
  int sign = 1;
  int i = 0;

  // Handle whitespace
  while(str[i] == ' ' || str[i] == '\t' || str[i] == '\n')
  {
    i++;
  }

  // Handle sign
  if(str[i] == '-')
  {
    sign = -1;
    i++;
  }
  else if(str[i] == '+')
  {
    i++;
  }

  // Handle base
  if(base == 0)
  {
    if(str[i] == '0')
    {
      if(str[i + 1] == 'x' || str[i + 1] == 'X')
      {
        base = 16;
        i += 2;
      }
      else
      {
        base = 8;
        i++;
      }
    }
    else
    {
      base = 10;
    }
  }
  else if(
    base == 16 && str[i] == '0' && (str[i + 1] == 'x' || str[i + 1] == 'X'))
  {
    i += 2;
  }

  // Convert digits
  while((str[i] >= '0' && str[i] <= '9') ||
        (base == 16 && str[i] >= 'a' && str[i] <= 'f') ||
        (base == 16 && str[i] >= 'A' && str[i] <= 'F'))
  {
    if(result > (LONG_MAX / base))
      return sign == -1 ? LONG_MIN : LONG_MAX;
    result = result * base;
    int digit = str[i] >= 'a'   ? str[i] - 'a' + 10
                : str[i] >= 'A' ? str[i] - 'A' + 10
                                : str[i] - '0';
    result += digit;
    i++;
  }

  // Set end pointer
  if(endptr != NULL)
  {
    *endptr = (char *)&str[i];
  }

  return sign * result;
}
#endif

int atoi(const char *nptr)
{
__ESBMC_HIDE:;
  return (int)strtol(nptr, (char **)0, 10);
}

long atol(const char *nptr)
{
__ESBMC_HIDE:;
  return strtol(nptr, (char **)0, 10);
}

char *getenv(const char *name)
{
__ESBMC_HIDE:;

  _Bool found;
  if(!found)
    return 0;

  char *buffer;
  size_t buf_size;

  __ESBMC_assume(buf_size >= 1);
  buffer = (char *)malloc(buf_size);
  buffer[buf_size - 1] = 0;
  return buffer;
}

typedef unsigned int gfp_t;

void *__kmalloc(size_t size, gfp_t flags)
{
  (void)flags;
  return malloc(size);
}

void *kmalloc(size_t size, gfp_t flags)
{
  (void)flags;
  return malloc(size);
}

void *kzalloc(size_t size, gfp_t flags)
{
  (void)flags;
  return malloc(size);
}

void *ldv_malloc(size_t size)
{
  return malloc(size);
}

void *ldv_zalloc(size_t size)
{
  return malloc(size);
}

void *kmalloc_array(size_t n, size_t size, gfp_t flags)
{
  return __kmalloc(n * size, flags);
}

void *kcalloc(size_t n, size_t size, gfp_t flags)
{
  (void)flags;
  return calloc(n, size);
}

void kfree(void *objp)
{
  free(objp);
}

size_t strlcat(char *dst, const char *src, size_t siz)
{
  char *d = dst;
  const char *s = src;
  size_t n = siz;
  size_t dlen;

  /* Find the end of dst and adjust bytes left but don't go past end */
  while(n-- != 0 && *d != '\0')
    d++;
  dlen = d - dst;
  n = siz - dlen;

  if(n == 0)
    return (dlen + strlen(s));
  while(*s != '\0')
  {
    if(n != 1)
    {
      *d++ = *s;
      n--;
    }
    s++;
  }
  *d = '\0';

  return (dlen + (s - src)); /* count does not include NUL */
}
