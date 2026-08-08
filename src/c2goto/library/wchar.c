#include <wchar.h>

// Wide-character models. Each mirrors the narrow-character model in string.c:
// a plain loop, so the bodies are exact rather than nondet and a caller that
// reads the destination afterwards sees what was written.

size_t wcslen(const wchar_t *s)
{
__ESBMC_HIDE:;
  size_t len = 0;
  while (s[len] != 0)
    len++;
  return len;
}

int wcscmp(const wchar_t *s1, const wchar_t *s2)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (s1[i] != 0 && s1[i] == s2[i])
    i++;
  if (s1[i] == s2[i])
    return 0;
  return s1[i] < s2[i] ? -1 : 1;
}

int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (i < n && s1[i] != 0 && s1[i] == s2[i])
    i++;
  if (i == n || s1[i] == s2[i])
    return 0;
  return s1[i] < s2[i] ? -1 : 1;
}

wchar_t *wcscpy(wchar_t *dst, const wchar_t *src)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (src[i] != 0)
  {
    dst[i] = src[i];
    i++;
  }
  dst[i] = 0;
  return dst;
}

wchar_t *wcsncpy(wchar_t *dst, const wchar_t *src, size_t n)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (i < n && src[i] != 0)
  {
    dst[i] = src[i];
    i++;
  }
  // strncpy pads the remainder with nulls, and so does wcsncpy.
  while (i < n)
  {
    dst[i] = 0;
    i++;
  }
  return dst;
}

wchar_t *wcscat(wchar_t *dst, const wchar_t *src)
{
__ESBMC_HIDE:;
  size_t d = wcslen(dst);
  size_t i = 0;
  while (src[i] != 0)
  {
    dst[d + i] = src[i];
    i++;
  }
  dst[d + i] = 0;
  return dst;
}

wchar_t *wcschr(const wchar_t *s, wchar_t c)
{
__ESBMC_HIDE:;
  size_t i = 0;
  while (s[i] != 0)
  {
    if (s[i] == c)
      return (wchar_t *)&s[i];
    i++;
  }
  // The terminator is part of the string for search purposes.
  return c == 0 ? (wchar_t *)&s[i] : (wchar_t *)0;
}

wchar_t *wcsrchr(const wchar_t *s, wchar_t c)
{
__ESBMC_HIDE:;
  const wchar_t *found = (const wchar_t *)0;
  size_t i = 0;
  while (s[i] != 0)
  {
    if (s[i] == c)
      found = &s[i];
    i++;
  }
  if (c == 0)
    return (wchar_t *)&s[i];
  return (wchar_t *)found;
}

int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < n; i++)
    if (s1[i] != s2[i])
      return s1[i] < s2[i] ? -1 : 1;
  return 0;
}

wchar_t *wmemcpy(wchar_t *dst, const wchar_t *src, size_t n)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < n; i++)
    dst[i] = src[i];
  return dst;
}

wchar_t *wmemmove(wchar_t *dst, const wchar_t *src, size_t n)
{
__ESBMC_HIDE:;
  // Copy in the direction that survives an overlap.
  if (dst < src)
  {
    for (size_t i = 0; i < n; i++)
      dst[i] = src[i];
  }
  else
  {
    for (size_t i = n; i > 0; i--)
      dst[i - 1] = src[i - 1];
  }
  return dst;
}

wchar_t *wmemset(wchar_t *s, wchar_t c, size_t n)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < n; i++)
    s[i] = c;
  return s;
}

wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n)
{
__ESBMC_HIDE:;
  for (size_t i = 0; i < n; i++)
    if (s[i] == c)
      return (wchar_t *)&s[i];
  return (wchar_t *)0;
}

int mbsinit(const mbstate_t *ps)
{
__ESBMC_HIDE:;
  // A null pointer denotes the initial state, and the model never leaves it.
  // Read the object representation rather than a member: mbstate_t's layout is
  // implementation-defined, and the UCRT's struct _Mbstatet -- which this
  // header defers to on Windows -- shares no member name with the fallback
  // declared for the other libcs. A zero-valued mbstate_t always describes an
  // initial conversion state (C11 7.29.1).
  if (ps == 0)
    return 1;
  const unsigned char *state = (const unsigned char *)ps;
  for (size_t i = 0; i < sizeof(mbstate_t); i++)
    if (state[i] != 0)
      return 0;
  return 1;
}
