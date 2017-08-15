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
