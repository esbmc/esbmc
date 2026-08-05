#ifndef __ESBMC_WCHAR_H
#define __ESBMC_WCHAR_H

#define __need_size_t
#define __need_wchar_t
#define __need_wint_t
#define __need_NULL
#include <stddef.h>

// Conversion state for the multibyte routines. The models below never leave
// the initial state -- they do not implement a shift encoding -- so a single
// flag is the whole representation.
typedef struct
{
  int __count;
} mbstate_t;

#ifndef WEOF
#  define WEOF ((wint_t)-1)
#endif

#ifndef WCHAR_MIN
#  define WCHAR_MIN 0
#endif
#ifndef WCHAR_MAX
#  define WCHAR_MAX 0x7fff
#endif

// Only the routines with a model in library/wchar.c are declared. Declaring a
// mutating routine without one would be worse than leaving it out: the call
// would return nondet and write nothing, so a later read of the destination
// sees stale data and can be proved correct when it is not. Add the rest here
// alongside their models.
size_t wcslen(const wchar_t *s);
int wcscmp(const wchar_t *s1, const wchar_t *s2);
int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n);
wchar_t *wcscpy(wchar_t *dst, const wchar_t *src);
wchar_t *wcsncpy(wchar_t *dst, const wchar_t *src, size_t n);
wchar_t *wcscat(wchar_t *dst, const wchar_t *src);
wchar_t *wcschr(const wchar_t *s, wchar_t c);
wchar_t *wcsrchr(const wchar_t *s, wchar_t c);
int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n);
wchar_t *wmemcpy(wchar_t *dst, const wchar_t *src, size_t n);
wchar_t *wmemmove(wchar_t *dst, const wchar_t *src, size_t n);
wchar_t *wmemset(wchar_t *s, wchar_t c, size_t n);
wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n);
int mbsinit(const mbstate_t *ps);

#endif
