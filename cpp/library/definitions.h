#ifndef STL_DEFINITIONS
#define STL_DEFINITIONS

#include <cstddef>

//#ifndef EOF
//#define EOF (-1)
//#endif

//#undef NULL
//#if defined(__cplusplus)
//#define NULL 0
//#else
//#define NULL ((void *)0)
//#endif

#define SIGINT  2

//#define EXIT_SUCCESS 0

#ifndef __TIMESTAMP__
#define __TIMESTAMP__ (0)
#endif

#ifdef _WIN64
    typedef __int64 streamsize;
#else
    typedef unsigned int streamsize;
#endif

unsigned int nondet_uint();
int nondet_int();
float nondet_float();
double nondet_double();
long double nondet_ldouble();
bool nondet_bool();
char* nondet_charPointer();
char nondet_char();
class smanip {};

//int sprintf ( char * str, const char * format, ... );

//#ifndef _size_t
//typedef unsigned int size_t;
//#endif

//#ifndef _ptrdiff_t
//typedef int ptrdiff_t;
//#endif

#define _SIZE_T_DEFINED

#endif
