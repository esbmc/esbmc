/* stddef.h is supposed to contain various compiler specific types and
 * facilities: */

typedef long int ptrdiff_t;

typedef short wchar_t;

#if __WORDSIZE == 64
typedef unsigned long int size_t;
#else
typedef unsigned long long int size_t;
#endif

#define NULL ((void *)0)

/* ESBMC's ANSI-C parser handles this natively */
#define offsetof(type, member) __builtin_offsetof(type, member)
