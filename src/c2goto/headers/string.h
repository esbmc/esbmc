
#pragma once

#include <__esbmc/stddefs.h>

#include <stddef.h> /* size_t */

__ESBMC_C_CPP_BEGIN

char *strcpy(char *__ESBMC_restrict dst, const char *__ESBMC_restrict src);
char *strncpy(char *__ESBMC_restrict dst, const char *__ESBMC_restrict src, size_t n);
char *strcat(char *__ESBMC_restrict dst, const char *__ESBMC_restrict src);
char *strncat(char *__ESBMC_restrict dst, const char *__ESBMC_restrict src, size_t n);
size_t strlen(const char *s);
int strcmp(const char *p1, const char *p2);
int strncmp(const char *s1, const char *s2, size_t n);
char *strchr(const char *s, int ch);
char *strrchr(const char *s, int c);
size_t strspn(const char *s, const char *accept);
size_t strcspn(const char *s, const char *reject);
char *strpbrk(const char *s, const char *accept);
char *strstr(const char *str1, const char *str2);
char *strtok(char *__ESBMC_restrict str, const char *__ESBMC_restrict delim);
char *strdup(const char *str);
char *strerror(int errnum);
void *memcpy(void *__ESBMC_restrict dst, const void *__ESBMC_restrict src, size_t n);
void *__memcpy_impl(void *__ESBMC_restrict dst, const void *__ESBMC_restrict src, size_t n);
void *memset(void *s, int c, size_t n);
void *memmove(void *dest, const void *src, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
void *memchr(const void *buf, int ch, size_t n);
size_t strxfrm (char *__ESBMC_restrict dst, const char *__ESBMC_restrict src, size_t n);
int strcoll (const char *s1, const char *s2);

__ESBMC_C_CPP_END
