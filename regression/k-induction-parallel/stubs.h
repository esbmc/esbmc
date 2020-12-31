#ifndef _STUBS_H
#define _STUBS_H

#include "base.h"

#define NULL ((void *)0)
#define EOS 0
#define EOF -1
#define ERR -1

/* I had size_t being an unsigned long before, but that led to the
 * infamous "Equality without matching types" error when I used a
 * size_t to index into an array. */
typedef int size_t;
typedef int bool;
#define true 1
#define false 0

#define TYPECAST_MEMCPY 0

char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *haystack, const char *needle);
char *strncpy (char *dest, const char *src, size_t n);
char *strncpy_ptr (char *dest, const char *src, size_t n);
char *strcpy (char *dest, const char *src);
unsigned strlen(const char *s);
int strncmp (const char *s1, const char *s2, size_t n);
int strcmp (const char *s1, const char *s2);
char *strcat(char *dest, const char *src);

void *memcpy(void *dest, const void *src, size_t n);

int isascii (int c);
int isspace (int c);

int getc (/* ignore FILE* arg */);

/* Extensions to libc's string library */
char *strrand (char *s);
int istrrand (char *s);
int istrchr(const char *s, int c);
int istrrchr(const char *s, int c);
int istrncmp (const char *s1, int start, const char *s2, size_t n);
int istrstr(const char *haystack, const char *needle);

/* Hackish duplicate functions to enable us to determine which claims
 * are relevant. Oh, the hilarity. */
char *r_strncpy (char *dest, const char *src, size_t n);
char *r_strcpy (char *dest, const char *src);
char *r_strcat(char *dest, const char *src);
char *r_strncat(char *dest, const char *src, size_t n);
void *r_memcpy(void *dest, const void *src, size_t n);

#endif
