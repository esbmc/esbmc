#ifndef __ESBMC_HEADERS_STDBOOL_H_
#define __ESBMC_HEADERS_STDBOOL_H_

#ifndef __cplusplus
/* C11 7.18 requires bool to expand to _Bool. Typedef'ing it to int skipped the
 * conversion to 0 or 1 of C11 6.3.1.2, so `(bool)2 == true` was false. C23
 * makes the three names predefined, so the header must not redefine them. */
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ <= 201710L
#define bool _Bool
#define false 0
#define true 1
#endif
#endif

#define __bool_true_false_are_defined 1

#endif /* __ESBMC_HEADERS_STDBOOL_H_ */
