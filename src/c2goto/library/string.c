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
  __ESBMC_assert(src != ((char *)0), "Source pointer is null");

  size_t i = 0;
  size_t __j;
  // Invariant: every index already written matches src and was not the
  // terminator (the loop breaks as soon as it copies '\0').
  __ESBMC_loop_invariant(
      __ESBMC_forall(&__j, !(__j < i)
                            || (dst[__j] == src[__j] && src[__j] != '\0')));
  // __contractor_loop: strcpy:0
  for (;; ++i)
  {
    dst[i] = src[i];
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
  size_t __j;
  // Invariant: every index < copied is bound to a non-terminator byte that
  // already matches src, and copied is within [0, n].
  __ESBMC_loop_invariant(
      copied <= n
      && __ESBMC_forall(&__j, !(__j < copied)
                                || (dst[__j] == src[__j] && src[__j] != '\0')));
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
  size_t end = 0;
  size_t __j;
  // Invariant: every byte before end is non-null, so end indexes the
  // (so-far) earliest possible terminator of the existing dst string.
  __ESBMC_loop_invariant(
      __ESBMC_forall(&__j, !(__j < end) || start[__j] != '\0'));
  // __contractor_loop: strncat:0
  while (start[end] != '\0')
    ++end;

  size_t i = 0;
  // Invariant: i bytes have been copied from src into start[end..]; each
  // copied byte was non-null and matches src.
  __ESBMC_loop_invariant(
      i <= n
      && __ESBMC_forall(&__j, !(__j < i)
                                || (start[end + __j] == src[__j]
                                    && src[__j] != '\0')));
  // __contractor_loop: strncat:1
  while (i < n && src[i] != '\0')
  {
    start[end + i] = src[i];
    ++i;
  }
  start[end + i] = '\0';
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
  size_t __j;
  // Invariant: every byte before len is non-null. Combined with the loop
  // exit condition this gives the strlen post-condition directly.
  __ESBMC_loop_invariant(
      __ESBMC_forall(&__j, !(__j < len) || s[__j] != '\0'));
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
  size_t i = 0;
  unsigned char c1 = 0, c2 = 0;
  size_t __j;
  // Invariant: every byte at index < i was equal in s1 and s2 and non-null.
  __ESBMC_loop_invariant(
      __ESBMC_forall(&__j,
          !(__j < i) || (s1[__j] == s2[__j] && s1[__j] != '\0')));
  // __contractor_loop: strcmp:0
  do
  {
    c1 = s1[i];
    c2 = s2[i];
    ++i;
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
  unsigned char ch1 = 0, ch2 = 0;
  size_t __j;
  // Invariant: every index already inspected matched between s1 and s2;
  // i never exceeds n.
  __ESBMC_loop_invariant(
      i <= n
      && __ESBMC_forall(&__j, !(__j < i) || s1[__j] == s2[__j]));
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
  size_t i = 0;
  size_t __j;
  // Invariant: every index < i was a non-null byte different from ch.
  __ESBMC_loop_invariant(
      __ESBMC_forall(&__j, !(__j < i)
                            || (s[__j] != '\0' && s[__j] != (char)ch)));
  // __contractor_loop: strchr:0
  while (s[i] && s[i] != (char)ch)
    ++i;
  if (s[i] == (char)ch)
    return (char *)(s + i);
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
  // Invariant: every occurrence of c found so far satisfies *found == c
  // (the post-condition of strrchr); the unroll provides the BMC bound,
  // since strchr's recursion-through-this-loop is bounded by strlen(s).
  __ESBMC_loop_invariant(found == ((char *)0) || *found == (char)c);
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
  size_t __j;
  __ESBMC_loop_invariant(
      i <= n
      && __ESBMC_forall(&__j, !(__j < i) || cdst[__j] == csrc[__j]));
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
  size_t i = 0;
  size_t __j;
  __ESBMC_loop_invariant(
      i <= n
      && __ESBMC_forall(&__j, !(__j < i) || sp[__j] == (char)c));
  for (; i < n; i++)
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
  size_t __j;
  if (dest - src >= n)
  {
    size_t i = 0;
    // No-overlap forward case: csrc bytes are never aliased by an
    // already-written cdest cell, so the current value of csrc[__j]
    // equals the pre-call value for any __j we have already copied.
    __ESBMC_loop_invariant(
        i <= n
        && __ESBMC_forall(&__j, !(__j < i) || cdest[__j] == csrc[__j]));
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
    // Overlapping backward case: writes go from high to low so the still-
    // unread tail csrc[i..n) holds the original values; bytes in [i, n)
    // have already received the correct (original) source values.
    __ESBMC_loop_invariant(i <= n);
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
  const unsigned char *sc1 = s1, *sc2 = s2;
  size_t i = 0;
  size_t total = n;
  size_t __j;
  // Invariant: every byte already inspected is equal between sc1 and sc2;
  // i never exceeds the original length n.
  __ESBMC_loop_invariant(
      i <= total
      && __ESBMC_forall(&__j, !(__j < i) || sc1[__j] == sc2[__j]));
  // __contractor_loop: memcmp:0
  while (i < total)
  {
    int res = (int)sc1[i] - (int)sc2[i];
    if (res != 0)
      return res;
    ++i;
  }
  return 0;
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
  const unsigned char *p = (const unsigned char *)buf;
  size_t i = 0;
  size_t total = n;
  size_t __j;
  // Invariant: every byte at index < i is different from ch; i bounded by n.
  __ESBMC_loop_invariant(
      i <= total
      && __ESBMC_forall(&__j, !(__j < i) || p[__j] != (unsigned char)ch));
  // __contractor_loop: memchr:0
  while (i < total && p[i] != (unsigned char)ch)
    ++i;

  return (i < total ? (void *)(p + i) : ((void *)0));
}
