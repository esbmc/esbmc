
#ifdef __clang__

#include_next <stdarg.h>

#else /* old frontend */

#ifndef __ESBMC_HEADERS_STDARG_H_
#define __ESBMC_HEADERS_STDARG_H_
/* Define standard macros; esbmc currently copes with gcc internal forms,
 * so we'll just replicate those */

#define _VA_LIST
#define _VA_LIST_DECLARED /* FreeBSD */

#define va_start(v,l)   __ESBMC_va_start(v,l)
#define va_end(v)       __ESBMC_va_end(v)
#define va_arg(v,l)     ((l)*(l *)__ESBMC_va_arg(v))
#define va_copy(d,s)    __ESBMC_va_copy(d,s)

typedef char *va_list;

void __ESBMC_va_start(va_list, ...);
void __ESBMC_va_end(va_list);
void * __ESBMC_va_arg(va_list);
void __ESBMC_va_copy(va_list, va_list);

#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST 1
typedef va_list __gnuc_va_list;
#endif

#endif /* __ESBMC_HEADERS_STDARG_H_ */

#endif /* !defined __clang__ */
