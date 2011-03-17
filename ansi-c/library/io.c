
/* FUNCTION: putchar */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int putchar(int c)
{
  _Bool error;
  __ESBMC_HIDE: printf("%c", c);
  return (error?-1:c);
}

/* FUNCTION: puts */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int puts(const char *s)
{
  _Bool error;
  int ret;
  __ESBMC_HIDE: printf("%s\n", s);
  if(error) ret=-1; else __ESBMC_assume(ret>=0);
  return ret;
}

/* FUNCTION: fopen */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

#ifndef __ESBMC_STDLIB_H_INCLUDED
#include <stdlib.h>
#define __ESBMC_STDLIB_H_INCLUDED
#endif

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

/* FUNCTION: fclose */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int fclose(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  free(stream);
  return return_value;
}

/* FUNCTION: fdopen */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

#ifndef __ESBMC_STDLIB_H_INCLUDED
#include <stdlib.h>
#define __ESBMC_STDLIB_H_INCLUDED
#endif

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

/* FUNCTION: fgets */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

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

/* FUNCTION: fread */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

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

/* FUNCTION: feof */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int feof(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: ferror */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int ferror(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: fileno */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int fileno(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: fputs */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

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

/* FUNCTION: fflush */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int fflush(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: fpurge */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

int fpurge(FILE *stream)
{
  // just return nondet
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: read */

#ifndef __ESBMC_UNISTD_H_INCLUDED
#include <unistd.h>
#define __ESBMC_UNISTD_H_INCLUDED
#endif

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

/* FUNCTION: fgetc */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int fgetc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: getc */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int getc(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: getchar */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int getchar()
{
  __ESBMC_HIDE:;
  int return_value;
  return return_value;
}

/* FUNCTION: getw */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int getw(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: fseek */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline int fseek(FILE *stream, long offset, int whence)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: ftell */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline long ftell(FILE *stream)
{
  __ESBMC_HIDE:;
  int return_value;
  *stream;
  return return_value;
}

/* FUNCTION: rewind */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

inline void rewind(FILE *stream)
{
  __ESBMC_HIDE:
  *stream;
}

/* FUNCTION: fwrite */

#ifndef __ESBMC_STDIO_H_INCLUDED
#include <stdio.h>
#define __ESBMC_STDIO_H_INCLUDED
#endif

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

