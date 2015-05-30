#include <string.h>
#include <stdlib.h>

#include "intrinsics.h"

// Because of macs,
#undef strcpy
#undef strncpy
#undef strcat
#undef strncat
#undef memcpy
#undef memset
#undef memmove

char *strcpy(char *dst, const char *src)
{
  __ESBMC_HIDE:;
  size_t i;
  for(i=0; src[i]!=0; i++)
    dst[i]=src[i];
  return dst;
}

char *strcpy_strabs(char *dst, const char *src)
{
  __ESBMC_HIDE:;
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strcpy zero-termination of 2nd argument");
  __ESBMC_assert(__ESBMC_buffer_size(dst)>__ESBMC_zero_string_length(src), "strcpy buffer overflow");
  dst[__ESBMC_zero_string_length(src)]=0;
  __ESBMC_is_zero_string(dst)=1;
  __ESBMC_zero_string_length(dst)=__ESBMC_zero_string_length(src);
  return dst;
}

char *strncpy(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:;
  size_t i=0;
  char ch;
  _Bool end;

  for(end=0; i<n; i++)
  {
    ch=end?0:src[i];
    dst[i]=ch;
    end=end || ch==(char)0;
  }
  return dst;
}

char *strncpy_strabs(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:;
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strncpy zero-termination of 2nd argument");
  __ESBMC_assert(__ESBMC_buffer_size(dst)>=n, "strncpy buffer overflow");
  __ESBMC_is_zero_string(dst)=__ESBMC_zero_string_length(src)<n;
  __ESBMC_zero_string_length(dst)=__ESBMC_zero_string_length(src);
  return dst;
}

char *strcat(char *dst, const char *src)
{
  __ESBMC_HIDE:
  // XXX - this has no body if string abstraction option is not enabled
  return dst;
}

char *strcat_strabs(char *dst, const char *src)
{
  __ESBMC_HIDE:
  size_t new_size;
  __ESBMC_assert(__ESBMC_is_zero_string(dst), "strcat zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(src), "strcat zero-termination of 2nd argument");
  new_size=__ESBMC_zero_string_length(dst)+__ESBMC_zero_string_length(src);
  __ESBMC_assert(__ESBMC_buffer_size(dst)>=new_size,
                   "strcat buffer overflow");
  size_t old_size=__ESBMC_zero_string_length(dst);
  //"  for(size_t i=0; i<__ESBMC_zero_string_length(src); i++)
  //"    dst[old_size+i];
  dst[new_size]=0;
  __ESBMC_is_zero_string(dst)=1;
  __ESBMC_zero_string_length(dst)=new_size;
  return dst;
}

char *strncat(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:
  // XXX - this has no body if string abstraction option isn't enabled
  return dst;
}

char *strncat_strabs(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:
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
  return dst;
}

int strcmp(const char *s1, const char *s2)
{
  __ESBMC_HIDE:;
  int retval;
  if(s1!=0 && s1==s2) return 0;
  // XXX - this does nothing useful if string abstraction isn't defined
  return retval;
}

int strcmp_strabs(const char *s1, const char *s2)
{
  __ESBMC_HIDE:;
  int retval;
  if(s1!=0 && s1==s2) return 0;
  __ESBMC_assert(__ESBMC_is_zero_string(s1), "strcmp zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(s2), "strcmp zero-termination of 2nd argument");
  if(__ESBMC_zero_string_length(s1) != __ESBMC_zero_string_length(s1)) __ESBMC_assume(retval!=0);
  return retval;
}

int strncmp(const char *s1, const char *s2, size_t n)
{
  __ESBMC_HIDE:
  if(s1!=0 && s1==s2) return 0;
  // XXX - does nothing useful if string abstraction option not on
  return 0;
}

int strncmp_strabs(const char *s1, const char *s2, size_t n)
{
  if(s1!=0 && s1==s2) return 0;
  __ESBMC_assert(__ESBMC_is_zero_string(s1) || __ESBMC_buffer_size(s1)>=n, "strncmp zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(s1) || __ESBMC_buffer_size(s2)>=n, "strncmp zero-termination of 2nd argument");
  // XXX - doesn't do anything useful /even if/ string abs is on
  return 0;
}

size_t strlen(const char *s)
{
  __ESBMC_HIDE:
  size_t len=0;
  while(s[len]!=0) len++;
  return len;
}

size_t strlen_strabs(const char *s)
{
  __ESBMC_HIDE:
  //__ESBMC_assert(__ESBMC_is_zero_string(s), "strlen zero-termination");
  return __ESBMC_zero_string_length(s);
}

char *strdup(const char *str)
{
  __ESBMC_HIDE:;
  size_t bufsz;
  bufsz=(strlen(str)+1)*sizeof(char);
  char *cpy=malloc(bufsz);
  if(cpy==((void *)0)) return 0;
  cpy=strcpy(cpy, str);
  return cpy;
}

char *strdup_strabs(const char *str)
{
  __ESBMC_HIDE:;
  size_t bufsz;
  bufsz=(strlen_strabs(str)+1)*sizeof(char);
  char *cpy=malloc(bufsz);
  if(cpy==((void *)0)) return 0;
  __ESBMC_assume(__ESBMC_buffer_size(cpy)==bufsz);
  cpy=strcpy_strabs(cpy, str);
  return cpy;
}

void *memcpy(void *dst, const void *src, size_t n)
{
  __ESBMC_HIDE:
  char *cdst = dst;
  const char *csrc = src;
  for(size_t i=0; i<n ; i++)
    cdst[i] = csrc[i];
  return dst;
}

void *memcpy_strabs(void *dst, const void *src, size_t n)
{
  __ESBMC_HIDE:
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
  return dst;
}

void *memset(void *s, int c, size_t n)
{
  __ESBMC_HIDE:
  char *sp=s;
  for(size_t i=0; i<n ; i++) sp[i]=c;
  return s;
}

void *memset_strabs(void *s, int c, size_t n)
{
  __ESBMC_HIDE:
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
  return s;
}

void *memmove(void *dest, const void *src, size_t n)
{
  __ESBMC_HIDE:
  char *cdest = dest;
  const char *csrc = src;
  if (dest-src >= n)
  {
    for(size_t i=0; i<n ; i++)
      cdest[i] = csrc[i];
  }
  else
  {
    for(size_t i=n; i>0 ; i--)
      cdest[i-1] = csrc[i-1];
  }
  return dest;
}

void *memmove_strabs(void *dest, const void *src, size_t n)
{
  __ESBMC_HIDE:
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
  return dest;
}

int memcmp(const void *s1, const void *s2, size_t n)
{
  __ESBMC_HIDE:;
  int res;
  const unsigned char *sc1=s1, *sc2=s2;
  for(; n!=0; n--)
  {
    res = (sc1++) - (sc2++);
    if (res != 0)
      return res;
  }
  return res;
}

int memcmp_strabs(const void *s1, const void *s2, size_t n)
{
  __ESBMC_HIDE:;
  int res;
  __ESBMC_assert(__ESBMC_buffer_size(s1)>=n, "memcmp buffer overflow of 1st argument");
  __ESBMC_assert(__ESBMC_buffer_size(s2)>=n, "memcmp buffer overflow of 2nd argument");
  // XXX - memcmp doesn't do anything here when strabs is enabled
  return res;
}
