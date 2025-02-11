
/* This override the corresponding Windows "ucrt" header */

#include <corecrt.h>

/* do not provide inline definitions of functions that we model separately */
#pragma push_macro("_CRT_FUNCTIONS_REQUIRED")
#undef _CRT_FUNCTIONS_REQUIRED
#define _CRT_FUNCTIONS_REQUIRED 0

#undef __CRT__NO_INLINE
#define __CRT__NO_INLINE /* Don't let mingw insert code */

#undef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES /* Define <math.h> macros M_LN2, etc. */

#include_next <corecrt_math.h>

#pragma pop_macro("_CRT_FUNCTIONS_REQUIRED")
