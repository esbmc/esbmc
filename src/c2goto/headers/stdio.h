#pragma once

#include <__esbmc/stddefs.h>
#include <stddef.h> /* size_t, NULL */
#include <stdarg.h> /* va_list */

/* glibc/Apple expose __int64_t via stdio.h; replicate since our bundled header skips that chain. */
#ifndef _WIN32
# if defined(__linux__) && defined(__LP64__)
typedef long __int64_t;
# else
__extension__ typedef long long __int64_t;
# endif
#endif

__ESBMC_C_CPP_BEGIN

/*
 * Opaque FILE type.  Must be a complete type so that sizeof(FILE) compiles
 * in io.c (e.g. malloc(sizeof(FILE))).  The exact layout is irrelevant for
 * verification — ESBMC never reads its fields.
 */
typedef struct __esbmc_file_t
{
  unsigned char __dummy[64];
} FILE;

/* Standard streams */
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

#define EOF (-1)
#define BUFSIZ 8192
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define FILENAME_MAX 4096
#define FOPEN_MAX 16
#define L_tmpnam 20

/* setvbuf buffering modes, C17 7.21.1 */
#define _IOFBF 0
#define _IOLBF 1
#define _IONBF 2

/* C17 7.21.1: an object type holding a stream's position. Opaque here; only
   fgetpos/fsetpos consume it. */
typedef long fpos_t;

/* printf family — plain declarations so ESBMC's do_function_call sees the
 * base_name "printf" and routes them through symex_printf rather than
 * leaving them uninterpreted (the Windows SDK expands printf to an inline
 * calling __stdio_common_vfprintf, which ESBMC has no model for). */
int printf(const char *__ESBMC_restrict format, ...);
int fprintf(FILE *__ESBMC_restrict stream, const char *__ESBMC_restrict format,
            ...);
int sprintf(char *__ESBMC_restrict str, const char *__ESBMC_restrict format,
            ...);
int snprintf(
  char *__ESBMC_restrict str,
  size_t size,
  const char *__ESBMC_restrict format,
  ...);
int dprintf(int fd, const char *__ESBMC_restrict format, ...);

int vprintf(const char *__ESBMC_restrict format, va_list ap);
int vfprintf(
  FILE *__ESBMC_restrict stream,
  const char *__ESBMC_restrict format,
  va_list ap);
int vsprintf(
  char *__ESBMC_restrict str,
  const char *__ESBMC_restrict format,
  va_list ap);
int vsnprintf(
  char *__ESBMC_restrict str,
  size_t size,
  const char *__ESBMC_restrict format,
  va_list ap);

/* scanf family */
int scanf(const char *__ESBMC_restrict format, ...);
int fscanf(
  FILE *__ESBMC_restrict stream,
  const char *__ESBMC_restrict format,
  ...);
int sscanf(
  const char *__ESBMC_restrict str,
  const char *__ESBMC_restrict format,
  ...);

/* File operations */
FILE *fopen(
  const char *__ESBMC_restrict pathname,
  const char *__ESBMC_restrict mode);
FILE *freopen(
  const char *__ESBMC_restrict pathname,
  const char *__ESBMC_restrict mode,
  FILE *__ESBMC_restrict stream);
FILE *fdopen(int fd, const char *mode);
FILE *fmemopen(void *buf, size_t size, const char *mode);
int fclose(FILE *stream);
int fflush(FILE *stream);
size_t fread(
  void *__ESBMC_restrict ptr,
  size_t size,
  size_t nmemb,
  FILE *__ESBMC_restrict stream);
size_t fwrite(
  const void *__ESBMC_restrict ptr,
  size_t size,
  size_t nmemb,
  FILE *__ESBMC_restrict stream);
int fseek(FILE *stream, long offset, int whence);
long ftell(FILE *stream);
void rewind(FILE *stream);
int fgetpos(FILE *__ESBMC_restrict stream, fpos_t *__ESBMC_restrict pos);
int fsetpos(FILE *stream, const fpos_t *pos);
int feof(FILE *stream);
int ferror(FILE *stream);
void clearerr(FILE *stream);
int fileno(FILE *stream);
int fpurge(FILE *stream);

/* Character I/O */
int fgetc(FILE *stream);
int fputc(int c, FILE *stream);
int getc(FILE *stream);
int putc(int c, FILE *stream);
int getchar(void);
int putchar(int c);
char *fgets(char *__ESBMC_restrict str, int n, FILE *__ESBMC_restrict stream);
int fputs(const char *__ESBMC_restrict str, FILE *__ESBMC_restrict stream);
char *gets(char *str);
int puts(const char *str);
int getw(FILE *stream);
int ungetc(int c, FILE *stream);

/* Misc */
int remove(const char *pathname);
int rename(const char *oldpath, const char *newpath);
FILE *tmpfile(void);
char *tmpnam(char *str);
void perror(const char *s);
void setbuf(FILE *__ESBMC_restrict stream, char *__ESBMC_restrict buf);
int setvbuf(
  FILE *__ESBMC_restrict stream,
  char *__ESBMC_restrict buf,
  int mode,
  size_t size);

__ESBMC_C_CPP_END
