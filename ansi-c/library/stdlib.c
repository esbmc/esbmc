#include <stdlib.h>
#undef exit
#undef abort
#undef calloc
#undef atoi
#undef atol
#undef getenv

#include "intrinsics.h"

inline void exit(int status)
{
  __ESBMC_assume(0);
}

inline void abort(void)
{
  __ESBMC_assume(0);
}

inline void *calloc(size_t nmemb, size_t size)
{
  __ESBMC_HIDE:;
  size_t total_size=nmemb*size;
  void *res = malloc(total_size);
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_is_zero_string(res);
  __ESBMC_zero_string_length(res)=0;
  //for(int i=0; i<nmemb*size; i++) res[i]=0;
  #else
  // there should be memset here
  //char *p=res;
  //for(int i=0; i<total_size; i++) p[i]=0;
  #endif
  return res;
}

inline int atoi(const char *nptr)
{
  __ESBMC_HIDE:;
  int res;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(nptr),
    "zero-termination of argument of atoi");
  #endif
  return res;
}

inline long atol(const char *nptr)
{
  __ESBMC_HIDE:;
  long res;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(nptr),
    "zero-termination of argument of atol");
  #endif
  return res;
}

inline char *getenv(const char *name)
{
  __ESBMC_HIDE:;

  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(name),
    "zero-termination of argument of getenv");
  #endif

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

inline void* calloc(size_t nmemb, size_t size)
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
