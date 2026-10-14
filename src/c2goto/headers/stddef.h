#ifdef __clang__

/* clang's <stddef.h> is documented as re-includable: a caller defines a
 * __need_* macro to pull in that one type, which is how <wchar.h> obtains
 * wint_t. A one-shot guard around this file makes every include after the
 * first a no-op and loses the requested type, so the guard covers only the
 * non-clang fallback and these names are set only for a plain include. */
#if !defined(__need_ptrdiff_t) && !defined(__need_size_t) &&                   \
  !defined(__need_wchar_t) && !defined(__need_NULL) &&                         \
  !defined(__need_STDDEF_H_misc) && !defined(__need_max_align_t) &&            \
  !defined(__need_offsetof) && !defined(__need_wint_t)
#define __need_ptrdiff_t	/* ptrdiff_t */
#define __need_size_t		/* size_t */
#define __need_wchar_t		/* wchar_t */
#define __need_NULL		/* NULL */
#define __need_STDDEF_H_misc	/* offsetof() and max_align_t */
#define __need_max_align_t   /* llvm 21: max_align_t */
#define __need_offsetof   /* llvm 21: offestof() */
#endif
#include_next <stddef.h>

#else
#ifndef __ESBMC_HEADERS_STDDEF_H_
#define __ESBMC_HEADERS_STDDEF_H_

/* stddef.h is supposed to contain various compiler specific types and
 * facilities: */

#ifndef _PTRDIFF_T_DEFINED
typedef long int ptrdiff_t;
#define _PTRDIFF_T_DEFINED
#endif

#ifndef __cplusplus
#ifndef _WCHAR_T_DEFINED
typedef short wchar_t;
#define _WCHAR_T_DEFINED
#endif
#endif

// Appease mingw
#ifdef __need_wint_t
typedef short wint_t;
#endif

#ifndef _SIZE_T_DEFINED
typedef unsigned int size_t;
#define _SIZE_T_DEFINED
#endif

#undef NULL
#if defined(__cplusplus)
#define NULL 0
#else
#define NULL ((void *)0)
#endif

/* ESBMC's ANSI-C parser handles this natively */
#define offsetof(type, member) __builtin_offsetof(type, member)

#endif /* __ESBMC_HEADERS_STDDEF_H_ */

#endif /* !defined(__clang__) */
