#ifdef _MINGW
#define _MT /* Don't define putchar/getc/getchar for us */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef _MSVC
#include <BaseTsd.h>
#define ssize_t SSIZE_T
#endif

#ifdef _MINGW
#undef feof
#undef ferror
#endif

int putchar(int c)
{
__ESBMC_HIDE:;
  _Bool error;
  printf("%c", c);
  return (error ? -1 : c);
}

int puts(const char *s)
{
__ESBMC_HIDE:;
  _Bool error;
  int ret;
  printf("%s\n", s);
  if(error)
    ret = -1;
  else
    __ESBMC_assume(ret >= 0);
  return ret;
}

FILE *fopen(const char *filename, const char *m)
{
__ESBMC_HIDE:;
  FILE *f = malloc(sizeof(FILE));
  return f;
}

int fclose(FILE *stream)
{
__ESBMC_HIDE:;
  free(stream);
  return nondet_int();
}

FILE *fdopen(int handle, const char *m)
{
__ESBMC_HIDE:;
  FILE *f = malloc(sizeof(FILE));
  return f;
}

char *fgets(char *str, int size, FILE *stream)
{
__ESBMC_HIDE:;
  _Bool error;
  return error ? 0 : str;
}

size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream)
{
__ESBMC_HIDE:;
  size_t nread;
  size_t bytes = nread * size;
  size_t i;
  __ESBMC_assume(nread <= nitems);

  for(i = 0; i < bytes; i++)
    ((char *)ptr)[i] = nondet_char();

  return nread;
}

int feof(FILE *stream)
{
  // just return nondet
  return nondet_int();
}

int ferror(FILE *stream)
{
  // just return nondet
  return nondet_int();
}

int fileno(FILE *stream)
{
  // just return nondet
  return nondet_int();
}

int fputs(const char *s, FILE *stream)
{
  // just return nondet
  return nondet_int();
}

int fflush(FILE *stream)
{
  // just return nondet
  return nondet_int();
}

int fpurge(FILE *stream)
{
  // just return nondet
  return nondet_int();
}

ssize_t read(int fildes, void *buf, size_t nbyte)
{
__ESBMC_HIDE:;
  ssize_t nread;
  size_t i;
  __ESBMC_assume(nread <= nbyte);

  for(i = 0; i < nbyte; i++)
    ((char *)buf)[i] = nondet_char();

  return nread;
}

int fgetc(FILE *stream)
{
__ESBMC_HIDE:;
  return nondet_int();
}

int getc(FILE *stream)
{
__ESBMC_HIDE:;
  return nondet_int();
}

int getchar()
{
__ESBMC_HIDE:;
  return nondet_int();
}

int getw(FILE *stream)
{
__ESBMC_HIDE:;
  return nondet_int();
}

int fseek(FILE *stream, long offset, int whence)
{
__ESBMC_HIDE:;
  return nondet_int();
}

long ftell(FILE *stream)
{
__ESBMC_HIDE:;
  return nondet_int();
}

void rewind(FILE *stream)
{
__ESBMC_HIDE:;
}

size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
__ESBMC_HIDE:;
  size_t nwrite;
  __ESBMC_assume(nwrite <= nitems);
  return nwrite;
}
