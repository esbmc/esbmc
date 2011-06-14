/* stddef.h is supposed to contain various compiler specific types and
 * facilities: */

typedef long int ptrdiff_t;

typedef uint16_t wchar_t;

#define NULL ((void *)0)

/* ESBMC's ANSI-C parser handles this natively */
#define offsetof(type, member) __builtin_offsetof(type, member)
