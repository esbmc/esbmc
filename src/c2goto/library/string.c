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

__ESBMC_contract
char *strcpy(char *dst, const char *src)
{
    __ESBMC_requires(dst != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dst) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= 1);
    /* glibc forbids overlap between [dst, dst+strlen(src)+1) and src;
       different objects covers the common case. */
    __ESBMC_requires(__ESBMC_POINTER_OBJECT(dst) != __ESBMC_POINTER_OBJECT(src));
    __ESBMC_ensures(__ESBMC_return_value == dst);
__ESBMC_HIDE:;
  // Ensure src pointer is non-null
  __ESBMC_assert(src != ((char *)0), "Source pointer is null");

  __ESBMC_unroll(9);
  // Constant propagation-friendly loop
  // __contractor_loop: strcpy:0
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

__ESBMC_contract
char *strncpy(char *dst, const char *src, size_t n)
{
    __ESBMC_requires(dst != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dst) >= n);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= 1);
    __ESBMC_requires(__ESBMC_POINTER_OBJECT(dst) != __ESBMC_POINTER_OBJECT(src));
    __ESBMC_ensures(__ESBMC_return_value == dst);
__ESBMC_HIDE:;
  char *start = dst;
  size_t copied = 0;

  __ESBMC_unroll(9);
  // __contractor_loop: strncpy:0
  while (copied < n && src[copied] != '\0')
  {
    dst[copied] = src[copied];
    copied++;
  }

  if (copied < n)
    memset(dst + copied, 0, n - copied);

  return start;
}

__ESBMC_contract
char *strcat(char *dst, const char *src)
{
    __ESBMC_requires(dst != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dst) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= 1);
    __ESBMC_requires(__ESBMC_POINTER_OBJECT(dst) != __ESBMC_POINTER_OBJECT(src));
    __ESBMC_ensures(__ESBMC_return_value == dst);
__ESBMC_HIDE:;
  strcpy(dst + strlen(dst), src);
  return dst;
}

__ESBMC_contract
char *strncat(char *dst, const char *src, size_t n)
{
    __ESBMC_requires(dst != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dst) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= 1);
    __ESBMC_requires(__ESBMC_POINTER_OBJECT(dst) != __ESBMC_POINTER_OBJECT(src));
    __ESBMC_ensures(__ESBMC_return_value == dst);
__ESBMC_HIDE:;
  char *start = dst;

  __ESBMC_unroll(9);
  // __contractor_loop: strncat:0
  while (*dst++)
    ;
  dst--;

  __ESBMC_unroll(9);
  // __contractor_loop: strncat:1
  while (n--)
    if (!(*dst++ = *src++))
      return start;

  *dst = '\0';
  return start;
}

__ESBMC_contract
size_t strlen(const char *s)
{
    size_t __k;
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    /* result is the index of the first '\0' in s. */
    __ESBMC_ensures(s[__ESBMC_return_value] == '\0');
    __ESBMC_ensures(__ESBMC_forall(&__k,
        !(__k < __ESBMC_return_value) || s[__k] != '\0'));
__ESBMC_HIDE:;
  size_t len = 0;
  __ESBMC_unroll(9);
  // __contractor_loop: strlen:0
  while (s[len] != 0)
    len++;
  return len;
}

__ESBMC_contract
int strcmp(const char *p1, const char *p2)
{
    __ESBMC_requires(p1 != ((void *)0));
    __ESBMC_requires(p2 != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(p1) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(p2) >= 1);
    __ESBMC_ensures(__ESBMC_return_value == 0
        || __ESBMC_return_value < 0 || __ESBMC_return_value > 0);
__ESBMC_HIDE:;
  const unsigned char *s1 = (const unsigned char *)p1;
  const unsigned char *s2 = (const unsigned char *)p2;
  unsigned char c1, c2;

  __ESBMC_unroll(9);
  // __contractor_loop: strcmp:0
  do
  {
    c1 = (unsigned char)*s1++;
    c2 = (unsigned char)*s2++;
    if (c1 == '\0')
      return c1 - c2;
  } while (c1 == c2);

  return c1 - c2;
}

__ESBMC_contract
int strncmp(const char *s1, const char *s2, size_t n)
{
    __ESBMC_requires(s1 != ((void *)0));
    __ESBMC_requires(s2 != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s1) >= n);
    __ESBMC_requires(__ESBMC_get_object_size(s2) >= n);
    __ESBMC_ensures(__ESBMC_return_value == 0
        || __ESBMC_return_value == 1 || __ESBMC_return_value == -1);
__ESBMC_HIDE:;
  size_t i = 0;
  unsigned char ch1, ch2;
  __ESBMC_unroll(9);
  // __contractor_loop: strncmp:0
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

__ESBMC_contract
char *strchr(const char *s, int ch)
{
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    /* Returns pointer into s where ch first occurs, or NULL if absent.
       (When ch == '\0', returns pointer to the terminator.) */
    __ESBMC_ensures(__ESBMC_return_value == ((char *)0)
        || *(char *)__ESBMC_return_value == (char)ch);
__ESBMC_HIDE:;
  __ESBMC_unroll(9);
  // __contractor_loop: strchr:0
  while (*s && *s != (char)ch)
    s++;
  if (*s == (char)ch)
    return (char *)s;
  return ((char *)0);
}

__ESBMC_contract
char *strrchr(const char *s, int c)
{
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    /* Returns pointer to the last occurrence of c in s, or NULL. */
    __ESBMC_ensures(__ESBMC_return_value == ((char *)0)
        || *(char *)__ESBMC_return_value == (char)c);
__ESBMC_HIDE:;
  const char *found, *p;

  c = (unsigned char)c;

  /* Since strchr is fast, we use it rather than the obvious loop.  */

  if (c == '\0')
    return strchr(s, '\0');

  found = ((char *)0);
  __ESBMC_unroll(9);
  // __contractor_loop: strrchr:0
  while ((p = strchr(s, c)) != ((char *)0))
  {
    found = p;
    s = p + 1;
  }

  return (char *)found;
}

__ESBMC_contract
size_t strspn(const char *s, const char *accept)
{
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(accept != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(accept) >= 1);
__ESBMC_HIDE:;
  const char *p;
  const char *a;
  size_t count = 0;

  __ESBMC_unroll(9);
  // __contractor_loop: strspn:0
  for (p = s; *p != '\0'; ++p)
  {
    __ESBMC_unroll(9);
    // __contractor_loop: strspn:1
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

__ESBMC_contract
size_t strcspn(const char *s, const char *reject)
{
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(reject != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(reject) >= 1);
__ESBMC_HIDE:;
  size_t count = 0;

  __ESBMC_unroll(9);
  // __contractor_loop: strcspn:0
  while (*s != '\0')
    if (strchr(reject, *s++) == ((char *)0))
      ++count;
    else
      return count;

  return count;
}

__ESBMC_contract
char *strpbrk(const char *s, const char *accept)
{
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(accept != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(accept) >= 1);
__ESBMC_HIDE:;
  __ESBMC_unroll(9);
  // __contractor_loop: strpbrk:0
  while (*s != '\0')
  {
    const char *a = accept;
    __ESBMC_unroll(9);
    // __contractor_loop: strpbrk:1
    while (*a != '\0')
      if (*a++ == *s)
        return (char *)s;
    ++s;
  }

  return ((char *)0);
}

__ESBMC_contract
char *strstr(const char *str1, const char *str2)
{
    __ESBMC_requires(str1 != ((void *)0));
    __ESBMC_requires(str2 != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(str1) >= 1);
    __ESBMC_requires(__ESBMC_get_object_size(str2) >= 1);
__ESBMC_HIDE:;
  char *cp = (char *)str1;
  char *s1, *s2;

  if (!*str2)
    return (char *)str1;

  __ESBMC_unroll(9);
  // __contractor_loop: strstr:0
  while (*cp)
  {
    s1 = cp;
    s2 = (char *)str2;

    __ESBMC_unroll(9);
    // __contractor_loop: strstr:1
    while (*s1 && *s2 && !(*s1 - *s2))
      s1++, s2++;
    if (!*s2)
      return cp;
    cp++;
  }

  return ((char *)0);
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

__ESBMC_contract
char *strdup(const char *str)
{
    __ESBMC_requires(str != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(str) >= 1);
    /* Returns either NULL (alloc failure) or a pointer to a heap copy. */
__ESBMC_HIDE:;
  size_t bufsz;
  bufsz = (strlen(str) + 1);
  char *cpy = (char *)malloc(bufsz * sizeof(char));
  if (cpy == ((void *)0))
    return ((char *)0);
  strcpy(cpy, str);
  return cpy;
}

void *__memcpy_impl(void *dst, const void *src, size_t n)
{
__ESBMC_HIDE:;
  char *cdst = dst;
  const char *csrc = src;
  size_t i = 0;
  __ESBMC_unroll(9);
  while (i < n)
  {
    cdst[i] = csrc[i];
    ++i;
  }
  return dst;
}

__ESBMC_contract
void *memcpy(void *dst, const void *src, size_t n)
{
    size_t __i;
    __ESBMC_requires(dst != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dst) >= n);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= n);
    /* glibc says memcpy is UB when [dst, dst+n) and [src, src+n) overlap.
       Two pointers don't overlap iff they point to different objects, or
       (within the same object) one range ends before the other begins. */
    __ESBMC_requires(__ESBMC_POINTER_OBJECT(dst) != __ESBMC_POINTER_OBJECT(src)
        || __ESBMC_POINTER_OFFSET(dst) + n <= __ESBMC_POINTER_OFFSET(src)
        || __ESBMC_POINTER_OFFSET(src) + n <= __ESBMC_POINTER_OFFSET(dst));
    __ESBMC_ensures(__ESBMC_return_value == dst);
    __ESBMC_ensures(__ESBMC_forall(&__i,
        !(__i < n) || ((unsigned char *)dst)[__i] == ((const unsigned char *)src)[__i]));
__ESBMC_HIDE:;
  void *hax = &__memcpy_impl;
  (void)hax;
  return __ESBMC_memcpy(dst, src, n);
}

void *__memset_impl(void *s, int c, size_t n)
{
__ESBMC_HIDE:;
  char *sp = s;
  __ESBMC_unroll(9);
  for (size_t i = 0; i < n; i++)
    sp[i] = c;
  return s;
}

__ESBMC_contract
void *memset(void *s, int c, size_t n)
{
    size_t __i;
    __ESBMC_requires(s != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s) >= n);
    __ESBMC_ensures(__ESBMC_return_value == s);
    __ESBMC_ensures(__ESBMC_forall(&__i,
        !(__i < n) || ((unsigned char *)s)[__i] == (unsigned char)c));
__ESBMC_HIDE:;
  void *hax = &__memset_impl;
  (void)hax;
  return __ESBMC_memset(s, c, n);
}

__ESBMC_contract
void *memmove(void *dest, const void *src, size_t n)
{
    size_t __i;
    __ESBMC_requires(dest != ((void *)0));
    __ESBMC_requires(src != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(dest) >= n);
    __ESBMC_requires(__ESBMC_get_object_size(src) >= n);
    __ESBMC_ensures(__ESBMC_return_value == dest);
    /* memmove explicitly allows overlap, so the post-call src bytes may
       differ from the pre-call values; the spec is "dest[i] equals the
       value src[i] held at function entry". */
    __ESBMC_ensures(__ESBMC_forall(&__i,
        !(__i < n) || ((unsigned char *)dest)[__i]
                      == __ESBMC_old(((const unsigned char *)src)[__i])));
__ESBMC_HIDE:;
  char *cdest = dest;
  const char *csrc = src;
  if (dest - src >= n)
  {
    size_t i = 0;
    __ESBMC_unroll(9);
    // __contractor_loop: memmove:0
    while (i < n)
    {
      cdest[i] = csrc[i];
      ++i;
    }
  }
  else
  {
    size_t i = n;
    __ESBMC_unroll(9);
    // __contractor_loop: memmove:1
    while (i > 0)
    {
      cdest[i - 1] = csrc[i - 1];
      --i;
    }
  }
  return dest;
}

__ESBMC_contract
int memcmp(const void *s1, const void *s2, size_t n)
{
    size_t __i;
    __ESBMC_requires(s1 != ((void *)0));
    __ESBMC_requires(s2 != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(s1) >= n);
    __ESBMC_requires(__ESBMC_get_object_size(s2) >= n);
    /* result == 0 iff all n bytes are equal (lexicographic equality). */
    __ESBMC_ensures(!__ESBMC_forall(&__i,
        !(__i < n) || ((const unsigned char *)s1)[__i]
                      == ((const unsigned char *)s2)[__i])
        || __ESBMC_return_value == 0);
    __ESBMC_ensures(__ESBMC_return_value != 0
        || __ESBMC_forall(&__i,
        !(__i < n) || ((const unsigned char *)s1)[__i]
                      == ((const unsigned char *)s2)[__i]));
__ESBMC_HIDE:;
  int res = 0;
  const unsigned char *sc1 = s1, *sc2 = s2;
  __ESBMC_unroll(9);
  // __contractor_loop: memcmp:0
  while (n != 0)
  {
    res = (*sc1++) - (*sc2++);
    if (res != 0)
      return res;
    n--;
  }
  return res;
}

__ESBMC_contract
void *memchr(const void *buf, int ch, size_t n)
{
    size_t __i;
    __ESBMC_requires(buf != ((void *)0));
    __ESBMC_requires(__ESBMC_get_object_size(buf) >= n);
    /* result == NULL iff no byte in [0, n) equals (unsigned char)ch. */
    __ESBMC_ensures(!__ESBMC_forall(&__i,
        !(__i < n) || ((const unsigned char *)buf)[__i] != (unsigned char)ch)
        || __ESBMC_return_value == ((void *)0));
    __ESBMC_ensures(__ESBMC_return_value != ((void *)0)
        || __ESBMC_forall(&__i,
        !(__i < n) || ((const unsigned char *)buf)[__i] != (unsigned char)ch));
__ESBMC_HIDE:;
  __ESBMC_unroll(9);
  // __contractor_loop: memchr:0
  while (n && (*(unsigned char *)buf != (unsigned char)ch))
  {
    buf = (unsigned char *)buf + 1;
    n--;
  }

  return (n ? (void *)buf : ((void *)0));
}
