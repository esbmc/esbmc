#ifndef STL_DEFINITIONS
#define STL_DEFINITIONS

#include "cstddef"

#define SIGINT  2

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

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
unsigned nondet_unsigned();
class smanip {};

#define _SIZE_T_DEFINED

#endif
