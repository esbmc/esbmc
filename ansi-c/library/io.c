#include <stdio.h>
#include <stdlib.h>

#include "intrinsics.h"

inline int putchar(int c)
{
  _Bool error;
  __ESBMC_HIDE: printf("%c", c);
  return (error?-1:c);
}

inline int puts(const char *s)
{
  _Bool error;
  int ret;
  __ESBMC_HIDE: printf("%s\n", s);
  if(error) ret=-1; else __ESBMC_assume(ret>=0);
  return ret;
}

inline FILE *fopen(const char *filename, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));

  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(f), "fopen zero-termination of 1st argument");
  __ESBMC_assert(__ESBMC_is_zero_string(m), "fopen zero-termination of 2nd argument");
  #endif

  return f;
}

inline int fclose(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  free(stream);
  return return_value;
}

inline FILE *fdopen(int handle, const char *m)
{
  __ESBMC_HIDE:;
  FILE *f=malloc(sizeof(FILE));

  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(m),
    "fdopen zero-termination of 2nd argument");
  #endif

  return f;
}

inline char *fgets(char *str, int size, FILE *stream)
{
  __ESBMC_HIDE:;
  _Bool error;

  #ifdef __ESBMC_STRING_ABSTRACTION
  int resulting_size;
  __ESBMC_assert(__ESBMC_buffer_size(str)>=size, "buffer-overflow in fgets");
  if(size>0)
  {
    __ESBMC_assume(resulting_size<size);
    __ESBMC_is_zero_string(str)=!error;
    __ESBMC_zero_string_length(str)=resulting_size;
  }
  #endif

  return error?0:str;
}

inline size_t fread(
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

inline int feof(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

inline int ferror(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

inline int fileno(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

inline int fputs(const char *s, FILE *stream)
{
  // just return nondet
  int return_value;
  #ifdef __ESBMC_STRING_ABSTRACTION
  __ESBMC_assert(__ESBMC_is_zero_string(s), "fputs zero-termination of 1st argument");
  #endif
  *stream;
  return return_value;
}

inline int fflush(FILE *stream)
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

inline ssize_t read(int fildes, void *buf, size_t nbyte)
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

inline int fgetc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

inline int getc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

inline int getchar()
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

inline int getw(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

inline int fseek(FILE *stream, long offset, int whence)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

inline long ftell(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

inline void rewind(FILE *stream)
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
