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
  __ESBMC_HIDE:;
  _Bool error;
  printf("%c", c);
  return (error?-1:c);
}

int puts(const char *s)
{
  __ESBMC_HIDE:;
  _Bool error;
  int ret;
  printf("%s\n", s);
  if(error) ret=-1; else __ESBMC_assume(ret>=0);
  return ret;
}

struct _IO_FILE *fopen(const char *filename, const char *m)
{
  __ESBMC_HIDE:;
  struct _IO_FILE *f=malloc(sizeof(struct _IO_FILE));
  return f;
}

int fclose(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  free(stream);
  return return_value;
}

struct _IO_FILE *fdopen(int handle, const char *m)
{
  __ESBMC_HIDE:;
  struct _IO_FILE *f=malloc(sizeof(struct _IO_FILE));

  return f;
}

char *fgets(char *str, int size, struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  _Bool error;
  return error?0:str;
}

size_t fread(
  void *ptr,
  size_t size,
  size_t nitems,
  struct _IO_FILE *stream)
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

int feof(struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
  return return_value;
}

int ferror(struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
  return return_value;
}

int fileno(struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
  return return_value;
}

int fputs(const char *s, struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
  /* XXX - what? */
  return return_value;
}

int fflush(struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
  return return_value;
}

int fpurge(struct _IO_FILE *stream)
{
  // just return nondet
  int return_value;
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

int fgetc(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

int getc(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

int getchar()
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

int getw(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

int fseek(struct _IO_FILE *stream, long offset, int whence)
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

long ftell(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

void rewind(struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
}

size_t fwrite(
  const void *ptr,
  size_t size,
  size_t nitems,
  struct _IO_FILE *stream)
{
  __ESBMC_HIDE:;
  size_t nwrite;
  __ESBMC_assume(nwrite<=nitems);
  return nwrite;
}
