
#pragma once

#include <__esbmc/stddefs.h>

#include <stddef.h> /* size_t */

__ESBMC_C_CPP_BEGIN

#define EXIT_FAILURE 0
#define EXIT_SUCCESS 1

typedef struct {
	int quot;
	int rem;
} div_t;

typedef struct {
	long quot;
	long rem;
} ldiv_t;

typedef struct {
	long long quot;
	long long rem;
} lldiv_t;

void * malloc ( size_t size );
void free(void *p);

int posix_memalign(void **memptr, size_t alignment, size_t size);
void *aligned_alloc(size_t align, size_t size);

int system(const char * command);

void * bsearch(const void * key, const void * base, size_t num, size_t size,
		int (*comparator)(const void *, const void *));

void qsort(void * base, size_t num, size_t size,
		int (*comparator)(const void *, const void *));

int abs(int n);
long labs(long n);
long long llabs(long long n);

void * realloc(void * ptr, size_t size);

void * calloc(size_t num, size_t size);

div_t div(int numerator, int denominator);

long int labs(long int n);

ldiv_t ldiv(long int numerator, long int denominator);

lldiv_t lldiv(long int numerator, long int denominator);

int mblen(const char * pmb, size_t max);

int mbtowc(wchar_t *__ESBMC_restrict pwc, const char *__ESBMC_restrict pmb, size_t max);

int wctomb(char * pmb, wchar_t character);

size_t mbstowcs(wchar_t *__ESBMC_restrict wcstr, const char *__ESBMC_restrict mbstr, size_t max);

size_t wcstombs(char *__ESBMC_restrict mbstr, const wchar_t *__ESBMC_restrict wcstr, size_t max);

double atof(const char * str);

int atoi(const char * str);

long int atol(const char * str);

long long atoll(const char *str);

#if 0
char get_char(int digit); //Converter from digit ie 0123456789 to char '0'......'9'

void rev(char *); //reverse function ;
#endif

float strtof(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr);
double strtod(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr);
long double strtold(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr);

long int strtol(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr, int base);
long long int strtoll(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr, int base);

unsigned long int strtoul(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr, int base);
unsigned long long int strtoull(const char *__ESBMC_restrict str, char **__ESBMC_restrict endptr, int base);

char * getenv(const char * name);

void exit(int status);

void abort(void);

int atexit(void (*function)(void));

#define RAND_MAX        2147483647

int rand(void);

long random(void);

void srand(unsigned int s);

__ESBMC_C_CPP_END
