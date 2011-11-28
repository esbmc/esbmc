#ifndef __ESBMC_HEADERS_STDDEF_H_
#define __ESBMC_HEADERS_STDDEF_H_
/* stddef.h is supposed to contain various compiler specific types and
 * facilities: */

typedef long int ptrdiff_t;

typedef short wchar_t;

typedef unsigned int size_t;

#define NULL ((void *)0)

/* ESBMC's ANSI-C parser handles this natively */
#define offsetof(type, member) __builtin_offsetof(type, member)

#endif /* __ESBMC_HEADERS_STDDEF_H_ */
