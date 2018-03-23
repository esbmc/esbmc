#include <stdlib.h>
#include <string.h>

#undef exit
#undef abort
#undef calloc
#undef atoi
#undef atol
#undef getenv

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-noreturn"
void exit(int status)
{
  __ESBMC_assume(0);
}

void abort(void)
{
  __ESBMC_assume(0);
}
#pragma clang diagnostic pop

void __VERIFIER_error(void)
{
ERROR:
  __ESBMC_assert(0, "Verifier error called\n");
}

void *calloc(size_t nmemb, size_t size)
{
__ESBMC_HIDE:;
  size_t total_size = nmemb * size;
  void *res = malloc(total_size);
  memset(res, 0, total_size);
  return res;
}

int atoi(const char *nptr)
{
__ESBMC_HIDE:;
  int res;
  /* XXX - does nothing without strabs */
  return res;
}

long atol(const char *nptr)
{
__ESBMC_HIDE:;
  long res;
  /* XXX - does nothing without strabs */
  return res;
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
