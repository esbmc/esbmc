
/* This override the corresponding Windows "ucrt" header */

/* do not provide inline definitions of functions that we model separately */
#undef _CRT_FUNCTIONS_REQUIRED
#define _CRT_FUNCTIONS_REQUIRED 0

#undef __CRT__NO_INLINE
#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include_next <corecrt_math.h>
