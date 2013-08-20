#ifndef _LIMITS_H___
#define _LIMITS_H___

#define CHAR_BIT __CHAR_BIT__

#define MB_LEN_MAX 1

#define SCHAR_MIN ((-__SCHAR_MAX__) - 1)
#define SCHAR_MAX __SCHAR_MAX__

#define UCHAR_MAX (255)

// Assume unsigned chars; fix in the future.
#define CHAR_MIN 0
#define CHAR_MAX UCHAR_MAX

#define SHRT_MAX __SHRT_MAX__
#define SHRT_MIN ((-__SHRT_MAX__) - 1)

#define USHRT_MAX (SHRT_MAX * 2U + 1U)

#define INT_MAX __INT_MAX__
#define INT_MIN (-INT_MAX - 1)

#define UINT_MAX (INT_MAX * 2U + 1U)

#define LONG_MAX __LONG_MAX__
#define LONG_MIN (-LONG_MAX - 1L)

/* Maximum value an `unsigned long int' can hold.  (Minimum is 0).  */
#define ULONG_MAX (LONG_MAX * 2UL + 1UL)

#define LLONG_MAX __LONG_LONG_MAX__
#define LLONG_MIN (-LLONG_MAX - 1LL)

#define ULLONG_MAX (LLONG_MAX * 2ULL + 1ULL)

#endif
