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
//
// C11 7.28 has <uchar.h> define mbstate_t too and ESBMC does not shadow that
// header, so both can reach one translation unit. Honour every libc's guard
// and set them all, so whichever header comes second stands down.
//
// The UCRT has no such guard: corecrt.h typedefs mbstate_t unconditionally,
// and every unshadowed UCRT header drags corecrt.h in. There is no race to
// win there, so defer to it -- as headers/corecrt_math.h already does.
#if __has_include(<corecrt.h>)
#  include <corecrt.h>
#elif !defined(__mbstate_t_defined) /* glibc */ &&                             \
  !defined(_MBSTATE_T) /* Darwin */ &&                                         \
  !defined(_MBSTATE_T_DECLARED) /* BSD */ &&                                   \
  !defined(__DEFINED_mbstate_t) /* musl */
#  define __mbstate_t_defined 1
#  define _MBSTATE_T
#  define _MBSTATE_T_DECLARED
#  define __DEFINED_mbstate_t
typedef struct
{
  int __count;
} mbstate_t;
#endif

#ifndef WEOF
#  define WEOF ((wint_t)-1)
#endif

// Take the limits from the target. <stdint.h> guards its own copies the same
// way, so a literal here would win on include order and disagree with wchar_t.
#ifndef WCHAR_MAX
#  define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
#  if L'\0' - 1 > 0
#    define WCHAR_MIN (L'\0' + 0)
#  else
#    define WCHAR_MIN (-WCHAR_MAX - 1)
#  endif
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
