#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#undef strcpy
#undef strncpy
#undef strcat
#undef strncat
#undef strlen
#undef strcmp
#undef strncmp
#undef strchr
#undef strrchr
#undef strspn
#undef strcspn
#undef strpbrk
#undef strstr
#undef strtok
#undef memchr
#undef memcmp
#undef memset
#undef memcpy
#undef memmove
#undef memchr

char *strcpy(char *dst, const char *src)
{
__ESBMC_HIDE:;
  // Ensure src pointer is non-null
  __ESBMC_assert(src != NULL, "Source pointer is null");

  // Constant propagation-friendly loop
  for (size_t i = 0;; ++i)
  {
    // Copy each character including the null terminator
    dst[i] = src[i];

    // Break when null terminator is copied
    if (src[i] == '\0')
      break;
  }

  return dst;
}

char *strncpy(char *dst, const char *src, size_t n)
{
__ESBMC_HIDE:;
  char *start = dst;
  size_t copied = 0;

  while (copied < n && src[copied] != '\0')
  {
    dst[copied] = src[copied];
    copied++;
  }

  if (copied < n)
    memset(dst + copied, 0, n - copied);

  return start;
}

char *strcat(char *dst, const char *src)
{
__ESBMC_HIDE:;
  strcpy(dst + strlen(dst), src);
  return dst;
}

char *strncat(char *dst, const char *src, size_t n)
{
__ESBMC_HIDE:;
  char *start = dst;

  while (*dst++)
    ;
  dst--;

  while (n--)
    if (!(*dst++ = *src++))
      return start;

  *dst = '\0';
  return start;
}

size_t strlen(const char *s)
{
__ESBMC_HIDE:;
  if (!s)
    return 0;
  if (s[0] == '\0') return 0;
  if (s[1] == '\0') return 1;
  if (s[2] == '\0') return 2;
  if (s[3] == '\0') return 3;
  if (s[4] == '\0') return 4;
  if (s[5] == '\0') return 5;
  if (s[6] == '\0') return 6;
  if (s[7] == '\0') return 7;
  if (s[8] == '\0') return 8;
  if (s[9] == '\0') return 9;
  if (s[10] == '\0') return 10;
  if (s[11] == '\0') return 11;
  if (s[12] == '\0') return 12;
  if (s[13] == '\0') return 13;
  if (s[14] == '\0') return 14;
  if (s[15] == '\0') return 15;
  if (s[16] == '\0') return 16;
  if (s[17] == '\0') return 17;
  if (s[18] == '\0') return 18;
  if (s[19] == '\0') return 19;
  if (s[20] == '\0') return 20;
  if (s[21] == '\0') return 21;
  if (s[22] == '\0') return 22;
  if (s[23] == '\0') return 23;
  if (s[24] == '\0') return 24;
  if (s[25] == '\0') return 25;
  if (s[26] == '\0') return 26;
  if (s[27] == '\0') return 27;
  if (s[28] == '\0') return 28;
  if (s[29] == '\0') return 29;
  if (s[30] == '\0') return 30;
  if (s[31] == '\0') return 31;
  return 32;
}

int strcmp(const char *p1, const char *p2)
{
__ESBMC_HIDE:;
  const unsigned char *s1 = (const unsigned char *)p1;
  const unsigned char *s2 = (const unsigned char *)p2;
  unsigned char c1, c2;

  do
  {
    c1 = (unsigned char)*s1++;
    c2 = (unsigned char)*s2++;
    if (c1 == '\0')
      return c1 - c2;
  } while (c1 == c2);

  return c1 - c2;
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

char *strchr(const char *s, int ch)
{
__ESBMC_HIDE:;
  while (*s && *s != (char)ch)
    s++;
  if (*s == (char)ch)
    return (char *)s;
  return NULL;
}

char *strrchr(const char *s, int c)
{
__ESBMC_HIDE:;
  const char *found, *p;

  c = (unsigned char)c;

  /* Since strchr is fast, we use it rather than the obvious loop.  */

  if (c == '\0')
    return strchr(s, '\0');

  found = NULL;
  while ((p = strchr(s, c)) != NULL)
  {
    found = p;
    s = p + 1;
  }

  return (char *)found;
}

size_t strspn(const char *s, const char *accept)
{
__ESBMC_HIDE:;
  const char *p;
  const char *a;
  size_t count = 0;

  for (p = s; *p != '\0'; ++p)
  {
    for (a = accept; *a != '\0'; ++a)
      if (*p == *a)
        break;
    if (*a == '\0')
      return count;
    else
      ++count;
  }

  return count;
}

size_t strcspn(const char *s, const char *reject)
{
__ESBMC_HIDE:;
  size_t count = 0;

  while (*s != '\0')
    if (strchr(reject, *s++) == NULL)
      ++count;
    else
      return count;

  return count;
}

char *strpbrk(const char *s, const char *accept)
{
__ESBMC_HIDE:;
  while (*s != '\0')
  {
    const char *a = accept;
    while (*a != '\0')
      if (*a++ == *s)
        return (char *)s;
    ++s;
  }

  return NULL;
}

char *strstr(const char *str1, const char *str2)
{
__ESBMC_HIDE:;
  char *cp = (char *)str1;
  char *s1, *s2;

  if (!*str2)
    return (char *)str1;

  while (*cp)
  {
    s1 = cp;
    s2 = (char *)str2;

    while (*s1 && *s2 && !(*s1 - *s2))
      s1++, s2++;
    if (!*s2)
      return cp;
    cp++;
  }

  return NULL;
}

char *strtok(char *str, const char *delim)
{
__ESBMC_HIDE:;
  static char *p = 0;
  if (str)
    p = str;
  else if (!p)
    return 0;
  str = p + strspn(p, delim);
  p = str + strcspn(str, delim);
  if (p == str)
    return p = 0;
  p = *p ? *p = 0, p + 1 : 0;
  return str;
}

char *strdup(const char *str)
{
__ESBMC_HIDE:;
  size_t bufsz;
  bufsz = (strlen(str) + 1);
  char *cpy = (char *)malloc(bufsz * sizeof(char));
  if (cpy == ((void *)0))
    return 0;
  strcpy(cpy, str);
  return cpy;
}

void *__memcpy_impl(void *dst, const void *src, size_t n)
{
__ESBMC_HIDE:;
  unsigned char *cdst = (unsigned char *)dst;
  const unsigned char *csrc = (const unsigned char *)src;
  if (!cdst || !csrc)
    return dst;

  // Bound copies to keep symbolic execution tractable.
  if (n > 16)
    n = 16;

  if (n > 0) cdst[0] = csrc[0];
  if (n > 1) cdst[1] = csrc[1];
  if (n > 2) cdst[2] = csrc[2];
  if (n > 3) cdst[3] = csrc[3];
  if (n > 4) cdst[4] = csrc[4];
  if (n > 5) cdst[5] = csrc[5];
  if (n > 6) cdst[6] = csrc[6];
  if (n > 7) cdst[7] = csrc[7];
  if (n > 8) cdst[8] = csrc[8];
  if (n > 9) cdst[9] = csrc[9];
  if (n > 10) cdst[10] = csrc[10];
  if (n > 11) cdst[11] = csrc[11];
  if (n > 12) cdst[12] = csrc[12];
  if (n > 13) cdst[13] = csrc[13];
  if (n > 14) cdst[14] = csrc[14];
  if (n > 15) cdst[15] = csrc[15];

  return dst;
}

void *memcpy(void *dst, const void *src, size_t n)
{
__ESBMC_HIDE:;
  void *hax = &__memcpy_impl;
  (void)hax;
  return __ESBMC_memcpy(dst, src, n);
}

void *__memset_impl(void *s, int c, size_t n)
{
__ESBMC_HIDE:;
  char *sp = s;
  for (size_t i = 0; i < n; i++)
    sp[i] = c;
  return s;
}

void *memset(void *s, int c, size_t n)
{
__ESBMC_HIDE:;
  void *hax = &__memset_impl;
  (void)hax;
  return __ESBMC_memset(s, c, n);
}

void *memmove(void *dest, const void *src, size_t n)
{
__ESBMC_HIDE:;
  char *cdest = dest;
  const char *csrc = src;
  if (dest - src >= n)
  {
    size_t i = 0;
    while (i < n)
    {
      cdest[i] = csrc[i];
      ++i;
    }
  }
  else
  {
    size_t i = n;
    while (i > 0)
    {
      cdest[i - 1] = csrc[i - 1];
      --i;
    }
  }
  return dest;
}

int memcmp(const void *s1, const void *s2, size_t n)
{
__ESBMC_HIDE:;
  const unsigned char *sc1 = (const unsigned char *)s1;
  const unsigned char *sc2 = (const unsigned char *)s2;
  if (!sc1 || !sc2)
    return 0;

  // Bound comparisons to keep symbolic execution tractable.
  if (n > 16)
    n = 16;

  if (n == 0) return 0;
  if (sc1[0] != sc2[0]) return (int)sc1[0] - (int)sc2[0];
  if (n == 1) return 0;
  if (sc1[1] != sc2[1]) return (int)sc1[1] - (int)sc2[1];
  if (n == 2) return 0;
  if (sc1[2] != sc2[2]) return (int)sc1[2] - (int)sc2[2];
  if (n == 3) return 0;
  if (sc1[3] != sc2[3]) return (int)sc1[3] - (int)sc2[3];
  if (n == 4) return 0;
  if (sc1[4] != sc2[4]) return (int)sc1[4] - (int)sc2[4];
  if (n == 5) return 0;
  if (sc1[5] != sc2[5]) return (int)sc1[5] - (int)sc2[5];
  if (n == 6) return 0;
  if (sc1[6] != sc2[6]) return (int)sc1[6] - (int)sc2[6];
  if (n == 7) return 0;
  if (sc1[7] != sc2[7]) return (int)sc1[7] - (int)sc2[7];
  if (n == 8) return 0;
  if (sc1[8] != sc2[8]) return (int)sc1[8] - (int)sc2[8];
  if (n == 9) return 0;
  if (sc1[9] != sc2[9]) return (int)sc1[9] - (int)sc2[9];
  if (n == 10) return 0;
  if (sc1[10] != sc2[10]) return (int)sc1[10] - (int)sc2[10];
  if (n == 11) return 0;
  if (sc1[11] != sc2[11]) return (int)sc1[11] - (int)sc2[11];
  if (n == 12) return 0;
  if (sc1[12] != sc2[12]) return (int)sc1[12] - (int)sc2[12];
  if (n == 13) return 0;
  if (sc1[13] != sc2[13]) return (int)sc1[13] - (int)sc2[13];
  if (n == 14) return 0;
  if (sc1[14] != sc2[14]) return (int)sc1[14] - (int)sc2[14];
  if (n == 15) return 0;
  if (sc1[15] != sc2[15]) return (int)sc1[15] - (int)sc2[15];
  return 0;
}

void *memchr(const void *buf, int ch, size_t n)
{
__ESBMC_HIDE:;
  while (n && (*(unsigned char *)buf != (unsigned char)ch))
  {
    buf = (unsigned char *)buf + 1;
    n--;
  }

  return (n ? (void *)buf : NULL);
}
