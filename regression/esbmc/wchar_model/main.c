// Wide-character models. ESBMC shipped no <wchar.h> at all, so any program
// using the wcs*/wmem* family failed to parse. Each model is a plain loop
// mirroring the narrow-character one in string.c, so the bodies are exact
// rather than nondet -- a caller reading the destination afterwards sees what
// was written (issue #5868).
#include <wchar.h>

int main(void)
{
  wchar_t src[4] = {'a', 'b', 'c', 0};
  wchar_t dst[4];

  wcscpy(dst, src);
  __ESBMC_assert(wcslen(dst) == 3, "wcslen counts to the terminator");
  __ESBMC_assert(wcscmp(dst, src) == 0, "a copy compares equal");
  __ESBMC_assert(dst[3] == 0, "the copy is terminated");

  wchar_t small[2] = {'a', 0};
  wchar_t padded[4];
  wcsncpy(padded, small, 4);
  __ESBMC_assert(
    padded[0] == 'a' && padded[1] == 0 && padded[3] == 0,
    "wcsncpy pads the remainder with nulls");

  __ESBMC_assert(wcschr(src, 'b') == &src[1], "wcschr finds a character");
  __ESBMC_assert(wcschr(src, 'z') == 0, "and reports an absent one");
  __ESBMC_assert(wcschr(src, 0) == &src[3], "the terminator is searchable");
  __ESBMC_assert(wcsrchr(src, 'c') == &src[2], "wcsrchr finds the last");

  wchar_t ov[4] = {'a', 'b', 'c', 'd'};
  wmemmove(&ov[1], &ov[0], 3);
  __ESBMC_assert(
    ov[1] == 'a' && ov[2] == 'b' && ov[3] == 'c',
    "wmemmove copies correctly when the ranges overlap");

  wchar_t f[3];
  wmemset(f, 'x', 3);
  __ESBMC_assert(f[0] == 'x' && f[2] == 'x', "wmemset fills");
  __ESBMC_assert(wmemchr(f, 'x', 3) == &f[0], "wmemchr finds");
  __ESBMC_assert(wmemchr(f, 'y', 3) == 0, "and reports absent");

  mbstate_t st = {0};
  __ESBMC_assert(mbsinit(&st), "a zeroed state is the initial one");

  // The limits must bound wchar_t itself: too narrow a value would still win
  // over <stdint.h>'s on include order, and check a program at the wrong width.
  __ESBMC_assert(
    (wchar_t)((long long)WCHAR_MAX + 1) != (long long)WCHAR_MAX + 1,
    "WCHAR_MAX is the largest wchar_t");
  __ESBMC_assert(
    (wchar_t)((long long)WCHAR_MIN - 1) != (long long)WCHAR_MIN - 1,
    "WCHAR_MIN is the smallest wchar_t");
  return 0;
}
