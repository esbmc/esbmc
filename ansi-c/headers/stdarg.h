#ifndef __ESBMC_HEADERS_STDARG_H_
#define __ESBMC_HEADERS_STDARG_H_
/* Define standard macros; esbmc currently copes with gcc internal forms,
 * so we'll just replicate those */

#define va_start(v,l)   __builtin_va_start(v,l)
#define va_end(v)       __builtin_va_end(v)
#define va_arg(v,l)     __builtin_va_arg(v,l)
#define va_copy(d,s)    __builtin_va_copy(d,s)
#define __GNUC_VA_LIST
#define __gnuc_va_list	__builtin_va_list
#define va_list		__builtin_va_list

#endif /* __ESBMC_HEADERS_STDARG_H_ */
