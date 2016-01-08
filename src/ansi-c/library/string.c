#include <string.h>
#include <stdlib.h>

#include "intrinsics.h"

// OSX headers,
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
  size_t i = 0;
  char ch;
  do
  {
    ch = src[i];
    dst[i] = ch;
    i++;
  } while (ch != (char) 0);
  return dst;
}

char *strncpy(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:;
  size_t i = 0;
  char ch;
  _Bool end;

  for (end = 0; i < n; i++)
  {
    ch = end ? 0 : src[i];
    dst[i] = ch;
    end = end || ch == (char) 0;
  }
  return dst;
}

char *strcat(char *dst, const char *src)
{
  __ESBMC_HIDE:;
  size_t i = 0;
  while (dst[i] != 0)
    i++;

  size_t j = 0;
  char ch;
  do
  {
    char ch = src[j];
    dst[i] = ch;
    i++;
    j++;
  } while (ch != (char) 0);
  return dst;
}

char *strncat(char *dst, const char *src, size_t n)
{
  __ESBMC_HIDE:;
  return dst;
}

int strcmp(const char *s1, const char *s2)
{
  __ESBMC_HIDE:;
  size_t i = 0;
  unsigned char ch1, ch2;
  do
  {
    ch1 = s1[i];
    ch2 = s2[i];

    if (ch1 == ch2)
    {
    }
    else if (ch1 < ch2)
      return -1;
    else
      return 1;

    i++;
  } while (ch1 != 0 && ch2 != 0);
  return 0;
}

int strncmp(const char *s1, const char *s2, size_t n)
{
  __ESBMC_HIDE:;
  size_t i = 0;
  unsigned char ch1, ch2;
  do
  {
    ch1 = s1[i];
    ch2 = s2[i];

    if (ch1 == ch2)
    {
    }
    else if (ch1 < ch2)
      return -1;
    else
      return 1;

    i++;
  } while (ch1 != 0 && ch2 != 0 && i < n);
  return 0;
}

size_t strlen(const char *s)
{
  __ESBMC_HIDE:;
  size_t len = 0;
  while (s[len] != 0)
    len++;
  return len;
}

char *strdup(const char *str)
{
  __ESBMC_HIDE:;
  size_t bufsz;
  bufsz = (strlen(str) + 1);
  char *cpy = (char *) malloc(bufsz * sizeof(char));
  if (cpy == ((void *) 0))
    return 0;
  strcpy(cpy, str);
  return cpy;
}

void *memcpy(void *dst, const void *src, size_t n)
{
  __ESBMC_HIDE:;
  char *cdst = dst;
  const char *csrc = src;
  for(size_t i=0; i<n ; i++)
    cdst[i] = csrc[i];
  return dst;
}

void *memset(void *s, int c, size_t n)
{
  __ESBMC_HIDE:;
  char *sp=s;
  for(size_t i=0; i<n ; i++) sp[i]=c;
  return s;
}

void *memmove(void *dest, const void *src, size_t n)
{
  __ESBMC_HIDE:;
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
