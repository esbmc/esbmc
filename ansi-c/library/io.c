#ifdef _MINGW
#define _MT /* Don't define putchar/getc/getchar for us */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include "intrinsics.h"

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
  _Bool error;
  __ESBMC_HIDE: printf("%c", c);
  return (error?-1:c);
}

int puts(const char *s)
{
  _Bool error;
  int ret;
  __ESBMC_HIDE: printf("%s\n", s);
  if(error) ret=-1; else __ESBMC_assume(ret>=0);
  return ret;
}

FILE *fopen(const char *filename, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));
  return f;
}

FILE *fopen_strabs(const char *filename, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));

  __ESBMC_assert(__ESBMC_is_zero_string(f), "fopen zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(m), "fopen zero-termination of 2nd argument");

  return f;
}

int fclose(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  free(stream);
  return return_value;
}

FILE *fdopen(int handle, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));

  return f;
}

FILE *fdopen_abs(int handle, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));

  __ESBMC_assert(__ESBMC_is_zero_string(m),
    "fdopen zero-termination of 2nd argument");

  return f;
}

char *fgets(char *str, int size, FILE *stream)
{
  __ESBMC_HIDE:;
  _Bool error;
  /* XXX - this does nothing without string abstraction option */
  return error?0:str;
}

char *fgets_strabs(char *str, int size, FILE *stream)
{
  __ESBMC_HIDE:;
  _Bool error;

  int resulting_size;
  __ESBMC_assert(__ESBMC_buffer_size(str)>=size, "buffer-overflow in fgets");
  if(size>0)
  {
    __ESBMC_assume(resulting_size<size);
    __ESBMC_is_zero_string(str)=!error;
    __ESBMC_zero_string_length(str)=resulting_size;
  }

  return error?0:str;
}

size_t fread(
  void *ptr,
  size_t size,
  size_t nitems,
  FILE *stream)
{
  __ESBMC_HIDE:;
  size_t nread;
  size_t bytes=nread*size;
  size_t i;
  __ESBMC_assume(nread<=nitems);

  for(i=0; i<bytes; i++)
  {
    char nondet_char;
    ((char *)ptr)[i]=nondet_char;
  }

  return nread;
}

int feof(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

int ferror(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

int fileno(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

int fputs(const char *s, FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  /* XXX - what? */
  return return_value;
}

int fputs_strabs(const char *s, FILE *stream)
{
  // just return nondet
  int return_value;
  __ESBMC_assert(__ESBMC_is_zero_string(s), "fputs zero-termination of 1st argument");
  *stream;
  return return_value;
}

int fflush(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

int fpurge(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

ssize_t read(int fildes, void *buf, size_t nbyte)
{
  __ESBMC_HIDE:;
  ssize_t nread;
  size_t i;
  __ESBMC_assume(nread<=nbyte);

  for(i=0; i<nbyte; i++)
  {
    char nondet_char;
    ((char *)buf)[i]=nondet_char;
  }

  return nread;
}

int fgetc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

int getc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

int getchar()
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

int getw(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

int fseek(FILE *stream, long offset, int whence)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

long ftell(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

void rewind(FILE *stream)
{
  __ESBMC_HIDE:
  *stream;
}

size_t fwrite(
  const void *ptr,
  size_t size,
  size_t nitems,
  FILE *stream)
{
  __ESBMC_HIDE:;
  size_t nwrite;
  __ESBMC_assume(nwrite<=nitems);
  return nwrite;
}
