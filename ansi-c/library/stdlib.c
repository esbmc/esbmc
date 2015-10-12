#include <stdlib.h>
#undef exit
#undef abort
#undef calloc
#undef atoi
#undef atol
#undef getenv

#include "intrinsics.h"

void exit(int status)
{
  __ESBMC_assume(0);
}

void abort(void)
{
  __ESBMC_assume(0);
}

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
  // there should be memset here
  //char *p=res;
  //for(int i=0; i<total_size; i++) p[i]=0;
  return res;
}

void *calloc_strabs(size_t nmemb, size_t size)
{
  __ESBMC_HIDE:;
  size_t total_size=nmemb*size;
  void *res = malloc(total_size);
  __ESBMC_is_zero_string(res);
  __ESBMC_zero_string_length(res)=0;
  return res;
}

int atoi(const char *nptr)
{
  __ESBMC_HIDE:;
  int res;
  /* XXX - does nothing without strabs */
  return res;
}

int atoi_strabs(const char *nptr)
{
  __ESBMC_HIDE:;
  int res;
  __ESBMC_assert(__ESBMC_is_zero_string(nptr),
    "zero-termination of argument of atoi");
  return res;
}

long atol(const char *nptr)
{
  __ESBMC_HIDE:;
  long res;
  /* XXX - does nothing without strabs */
  return res;
}

long atol_strabs(const char *nptr)
{
  __ESBMC_HIDE:;
  long res;
  __ESBMC_assert(__ESBMC_is_zero_string(nptr),
    "zero-termination of argument of atol");
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

char *getenv_strabs(const char *name)
{
  __ESBMC_HIDE:;

  __ESBMC_assert(__ESBMC_is_zero_string(name),
    "zero-termination of argument of getenv");

  _Bool found;
  if(!found) return 0;

  char *buffer;
  size_t buf_size;

  __ESBMC_assume(buf_size>=1);
  buffer=(char *)malloc(buf_size);
  buffer[buf_size-1]=0;
  return buffer;
}

#if 0
/* FUNCTION: calloc */

#ifndef __ESBMC_STDLIB_H_INCLUDED
#include <stdlib.h>
#define __ESBMC_STDLIB_H_INCLUDED
#endif

void* calloc(size_t nmemb, size_t size)
{
  __ESBMC_HIDE:;
  void *res = malloc(nmemb*size);
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_is_zero_string(res);
  __ESBMC_zero_string_length(res)=0;
  //for(int i=0; i<nmemb*size; i++) res[i]=0;
  #else
  char *p=res;
  for(int i=0; i<nmemb*size; i++) p[i]=0;
  #endif
  return res;
}
#endif
