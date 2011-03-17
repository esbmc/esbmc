/* FUNCTION: strcpy */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline char *strcpy(char *dst, const char *src)
{
  __ESBMC_HIDE:;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strcpy zero-termination of 2nd argument");
  __ESBMC_assert(__ESBMC_buffer_size(dst)>__ESBMC_zero_string_length(src), "strcpy buffer overflow");
  dst[__ESBMC_zero_string_length(src)]=0;
  __ESBMC_is_zero_string(dst)=1;
  __ESBMC_zero_string_length(dst)=__ESBMC_zero_string_length(src);
  #else
  size_t i;
  for(i=0; src[i]!=0; i++)
    dst[i]=src[i];
  #endif
  return dst;
}

/* FUNCTION: strncpy */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline char *strncpy(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strncpy zero-termination of 2nd argument");
  __ESBMC_assert(__ESBMC_buffer_size(dst)>=n, "strncpy buffer overflow");
  __ESBMC_is_zero_string(dst)=__ESBMC_zero_string_length(src)<n;
  __ESBMC_zero_string_length(dst)=__ESBMC_zero_string_length(src);
  #else
  size_t i=0;
  for( ; i<n && src[i]!=0; i++)
    dst[i]=src[i];
  for( ; i<n ; i++)
    dst[i]=0;
  #endif
  return dst;
}

/* FUNCTION: strcat */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline char *strcat(char *dst, const char *src)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  size_t new_size;
  __ESBMC_assert(__ESBMC_is_zero_string(dst), "strcat zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strcat zero-termination of 2nd argument");
  new_size=__ESBMC_zero_string_length(dst)+__ESBMC_zero_string_length(src);
  printf("new_size: %d", new_size);
  __ESBMC_assert(__ESBMC_buffer_size(dst)>=new_size,
                   "strcat buffer overflow");
  size_t old_size=__ESBMC_zero_string_length(dst);
  //"  for(size_t i=0; i<__ESBMC_zero_string_length(src); i++)
  //"    dst[old_size+i];
  dst[new_size - 1]=0;
  __ESBMC_is_zero_string(dst)=1;
  __ESBMC_zero_string_length(dst)=new_size;
  #endif
  return dst;
}

/* FUNCTION: strncat */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline char *strncat(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  size_t additional, new_size;
  __ESBMC_assert(__ESBMC_is_zero_string(dst), "strncat zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(src) || __ESBMC_buffer_size(src)>=n, "strncat zero-termination of 2nd argument");
  additional=(n<__ESBMC_zero_string_length(src))?n:__ESBMC_zero_string_length(src);
  new_size=__ESBMC_is_zero_string(dst)+additional;
  __ESBMC_assert(__ESBMC_buffer_size(dst)>new_size,
                   "strncat buffer overflow");
  size_t dest_len=__ESBMC_zero_string_length(dst);
  size_t i;
  for (i = 0 ; i < n && i<__ESBMC_zero_string_length(src) ; i++)
    dst[dest_len + i] = src[i];
  dst[dest_len + i] = 0;
  __ESBMC_is_zero_string(dst)=1;
  __ESBMC_zero_string_length(dst)=new_size;
  #endif
  return dst;
}

/* FUNCTION: strcmp */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline int strcmp(const char *s1, const char *s2)
{
  __ESBMC_HIDE:;
  int retval;
  if(s1!=0 && s1==s2) return 0;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(s1), "strcmp zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(s2), "strcmp zero-termination of 2nd argument");
  if(__ESBMC_zero_string_length(s1) != __ESBMC_zero_string_length(s1)) __ESBMC_assume(retval!=0);
  #endif
  return retval;
}

/* FUNCTION: strncmp */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline int strncmp(const char *s1, const char *s2, size_t n)
{
  __ESBMC_HIDE:
  if(s1!=0 && s1==s2) return 0;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(s1) || __ESBMC_buffer_size(s1)>=n, "strncmp zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(s1) || __ESBMC_buffer_size(s2)>=n, "strncmp zero-termination of 2nd argument");
  #else
  #endif
}

/* FUNCTION: strlen */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline size_t strlen(const char *s)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  //__ESBMC_assert(__ESBMC_is_zero_string(s), "strlen zero-termination");
  return __ESBMC_zero_string_length(s);
  #else
  size_t len=0;
  while(s[len]!=0) len++;
  return len;
  #endif
}

/* FUNCTION: strdup */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

#ifndef __ESBMC_STDLIB_H_INCLUDED
#include <stdlib.h>
#define __ESBMC_STDLIB_H_INCLUDED
#endif

inline char *strdup(const char *str)
{
  __ESBMC_HIDE:;
  size_t bufsz;
  bufsz=(strlen(str)+1)*sizeof(char);
  char *cpy=malloc(bufsz);
  if(cpy==((void *)0)) return 0;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assume(__ESBMC_buffer_size(cpy)==bufsz);
  #endif
  cpy=strcpy(cpy, str);
  return cpy;
}

/* FUNCTION: memcpy */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline void *memcpy(void *dst, const void *src, size_t n)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  //__ESBMC_assert(__ESBMC_buffer_size(src)>=n, "memcpy buffer overflow");
  //__ESBMC_assert(__ESBMC_buffer_size(dst)>=n, "memcpy buffer overflow");
  //  for(size_t i=0; i<n ; i++) dst[i]=src[i];
  if(__ESBMC_is_zero_string(src) &&
     n > __ESBMC_zero_string_length(src))
  {
    __ESBMC_is_zero_string(dst)=1;
    __ESBMC_zero_string_length(dst)=__ESBMC_zero_string_length(src);
  }
  else if(!(__ESBMC_is_zero_string(dst) &&
            n <= __ESBMC_zero_string_length(dst)))
    __ESBMC_is_zero_string(dst)=0;
  #else
  for(size_t i=0; i<n ; i++) dst[i]=src[i];
  #endif
  return dst;
}

/* FUNCTION: memset */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline void *memset(void *s, int c, size_t n)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  char *sp=s;
  for(size_t i=0; i<n ; i++) {sp[i]=c;}
#if 0
  __ESBMC_assert(__ESBMC_buffer_size(s) * sizeof(s) >= n, "memset buffer overflow");
  //for(size_t i=0; i<n ; i++) s[i]=c;

  if(__ESBMC_is_zero_string(s) &&
     n > __ESBMC_zero_string_length(s))
  {
    __ESBMC_is_zero_string(s)=1;
  }
  else if(c==0)
  {
    __ESBMC_is_zero_string(s)=1;
    __ESBMC_zero_string_length(s)=0;
  }
  else
    __ESBMC_is_zero_string(s)=0;
#endif
  #else
  char *sp=s;
  for(size_t i=0; i<n ; i++) sp[i]=c;
  #endif
  return s;
}

/* FUNCTION: memmove */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

inline void *memmove(void *dest, const void *src, size_t n)
{
  __ESBMC_HIDE:
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_buffer_size(src)>=n, "memmove buffer overflow");
  // dst = src (with overlap allowed)
  if(__ESBMC_is_zero_string(src) &&
     n > __ESBMC_zero_string_length(src))
  {
    __ESBMC_is_zero_string(src)=1;
    __ESBMC_zero_string_length(dest)=__ESBMC_zero_string_length(src);
  }
  else
    __ESBMC_is_zero_string(dest)=0;
  #else
  if (dest-src >= n)
  {
    for(size_t i=0; i<n ; i++) dest[i]=src[i];
  }
  else
  {
    for(size_t i=n; i>0 ; i--) dest[i-1]=src[i-1];
  }
  #endif
  return dest;
}

/* FUNCTION: memcmp */

#ifndef __ESBMC_STRING_H_INCLUDED
#include <string.h>
#define __ESBMC_STRING_H_INCLUDED
#endif

int memcmp(const void *s1, const void *s2, size_t n)
{
  __ESBMC_HIDE:;
  int res;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_buffer_size(s1)>=n, "memcmp buffer overflow of 1st argument");
  __ESBMC_assert(__ESBMC_buffer_size(s2)>=n, "memcmp buffer overflow of 2nd argument");
  #else
  const unsigned char *sc1=s1, *sc2=s2;
  for(; n!=0; n--)
  {
    res = (s1++) - (s2++);
    if (res != 0)
      return res;
  }
  #endif
  return res;
}
