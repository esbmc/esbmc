#ifdef _MINGW
#  define _MT /* Don't define putchar/getc/getchar for us */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef _MSVC
#  include <BaseTsd.h>
#  define ssize_t SSIZE_T
#endif

#ifdef _MINGW
#  undef feof
#  undef ferror
#endif

#undef putchar
#undef puts
#undef getc
#undef feof
#undef ferror
#undef fileno
#undef getchar

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
  if (error)
    ret = -1;
  else
    __ESBMC_assume(ret >= 0);
  return ret;
}

// Reads characters from the standard input (stdin)
// and stores them as a C string into str until a newline character
// or the end-of-file is reached.
// Source: https://www.cplusplus.com/reference/cstdio/gets/
char *gets(char *str)
{
__ESBMC_HIDE:;
  size_t i = 0;

  // produce a nondet number for the number of
  // characters to read from the standard input
  // which can lead to buffer overflows
  size_t size = nondet_uint();

  // add non-deterministic characters to str
  for (i = 0; i < size; i++)
  {
    // produce the non-deterministic character
    char character = nondet_char();
    // if the newline character is found, it is not copied into str
    if (character == '\n')
    {
      // append a terminating '\0' to the current position and return
      str[i] = '\0';
      return str;
    }
    // this should store the character
    str[i] = character;
  }

  // append a terminating '\0' to the last position and return
  str[size - 1] = '\0';

  return str;
}

FILE *fopen(const char *filename, const char *mode)
{
__ESBMC_HIDE:;

  if (filename == NULL || mode == NULL)
    return NULL; // On error, returns a null pointer.

  // Allocate memory for FILE structure
  FILE *f = malloc(sizeof(FILE)); // Dynamically allocate FILE structure
  if (f == NULL)
    return NULL; // On error, returns a null pointer.

  return f;
}

int fclose(FILE *stream)
{
__ESBMC_HIDE:;
  // Check for preconditions
  __ESBMC_assert(stream != NULL, "Stream must not be NULL");
  // Simulate possible success/failure
  _Bool success = nondet_bool(); // Returns 0 (success) or 1 (failure)
  __ESBMC_assume(
    success == 0 || success == 1); // Ensure only valid return values
  if (success)
  {
    free(stream);
    return 0;
  }
  else
    return EOF; // fclose failure
}


FILE *fdopen(int handle, const char *mode)
{
__ESBMC_HIDE:;
  // Validate file descriptor
  if (handle < 0)
    return NULL; // Invalid file descriptor

  // Validate mode
  if (mode == NULL || *mode == '\0')
    return NULL; // Invalid mode

  // Allocate memory for FILE structure
  FILE *f = malloc(sizeof(FILE));
  if (f == NULL)
    return NULL; // Not enough memory

  return f; // Successfully associated FILE* with handle
}

// fgets reads a line from the specified stream and stores it
// into the string pointed to by str. It stops when either (n-1)
// characters are read, the newline character is read, or the
// end-of-file is reached, whichever comes first.
// source: https://www.cplusplus.com/reference/cstdio/fgets/
char *fgets(char *str, int size, FILE *stream)
{
__ESBMC_HIDE:;
  // this is used to check for early loop termination
  _Bool early_termination = 0;
  // check for pre-conditions
  __ESBMC_assert(
    stream == stdin || stream != NULL,
    "the pointer to a file object must be a valid argument");

  //do nothing, report error
  if (size < 2)
    return NULL;

  int i = 0;
  // add non-deterministic characters to str
  for (i = 0; i <= (size - 1); i++)
  {
    // produce non-deterministic character
    int character = getc(stream);
    // stop if we get a new line character or an EOF
    if (character == '\n')
    {
      // A newline character makes fgets stop reading,
      // but it is considered a valid character by the function
      // and included in the string copied to str
      str[i] = '\n';
      early_termination = 1;
      break;
    }
    else if (character == EOF)
    {
      // we end the loop based on non-determinism
      early_termination = 1;
      break;
    }
    // this should store the character
    str[i] = character;
  }

  // has the loop terminated before filling in all positions?
  // note that the loop can stop at any iteration
  if (early_termination)
  {
    // didn't we reach the total buffer size?
    if (i < (size - 1))
    {
      // append a terminating '\0' to the next position and return
      str[++i] = '\0';
      return str;
    }
  }

  // append a terminating '\0' to the last position and return
  str[size - 1] = '\0';

  return str;
}

size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream)
{
__ESBMC_HIDE:;
  size_t nread;
  size_t bytes = nread * size;
  size_t i;
  __ESBMC_assume(nread <= nitems);

  for (i = 0; i < bytes; i++)
    ((char *)ptr)[i] = nondet_char();

  return nread;
}

int feof(FILE *stream)
{
__ESBMC_HIDE:;
  __ESBMC_assert(stream != NULL, "Ensure *stream is valid");
  // Returns either 0 (not EOF) or 1 (EOF reached)
  _Bool eof_state = nondet_bool();
  return eof_state;
}

int ferror(FILE *stream)
{
__ESBMC_HIDE:;
  __ESBMC_assert(stream != NULL, "Ensure *stream is valid");
  // Model the error state correctly using nondeterminism
  int error_state =
    nondet_bool(); // Returns either 0 (no error) or 1 (error detected)
  __ESBMC_assume(
    error_state == 0 || error_state == 1); // Ensures only valid outputs
  return error_state;
}

int fileno(FILE *stream)
{
__ESBMC_HIDE:;

  // Check for NULL or invalid stream
  if (stream == NULL)
    return -1; // Error: Invalid stream

  // Handle standard streams correctly
  if (stream == stdin)
    return 0; // STDIN_FILENO
  if (stream == stdout)
    return 1; // STDOUT_FILENO
  if (stream == stderr)
    return 2; // STDERR_FILENO

  // Return a nondeterministic file descriptor, but ensure it is non-negative
  int fd = nondet_int();
  if (fd < 0)
    return -1;

  return fd;
}

int fputs(const char *s, FILE *stream)
{
__ESBMC_HIDE:;
  // just return nondet
  return nondet_int();
}

int fflush(FILE *stream)
{
__ESBMC_HIDE:;
  // just return nondet
  return nondet_int();
}

int fpurge(FILE *stream)
{
__ESBMC_HIDE:;
  // just return nondet
  return nondet_int();
}

ssize_t read(int fildes, void *buf, size_t nbyte)
{
__ESBMC_HIDE:;
  ssize_t nread;
  size_t i;
  __ESBMC_assume(nread <= nbyte);

  for (i = 0; i < nbyte; i++)
    ((char *)buf)[i] = nondet_char();

  return nread;
}

int fgetc(FILE *stream)
{
__ESBMC_HIDE:;
  // Ensure the stream is valid
  __ESBMC_assert(stream != NULL, "fgetc: stream must not be NULL");
  // Nondeterministically return a character or EOF
  int c = nondet_int();
  // Ensure the returned value is either a valid unsigned char or EOF
  __ESBMC_assume(c == EOF || (c >= 0 && c <= 255));
  return c;
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
