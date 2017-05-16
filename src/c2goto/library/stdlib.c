#include <stdlib.h>
#include <string.h>

#undef exit
#undef abort
#undef calloc
#undef atoi
#undef atol
#undef getenv

#include "intrinsics.h"

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
  size_t total_size=nmemb*size;
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
  if(!found) return 0;

  char *buffer;
  size_t buf_size;

  __ESBMC_assume(buf_size>=1);
  buffer=(char *)malloc(buf_size);
  buffer[buf_size-1]=0;
  return buffer;
}
