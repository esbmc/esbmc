#ifdef _MINGW
#  define _MT /* Don't define putchar/getc/getchar for us */
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

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

static size_t __esbmc_buffer_pending = 0;
static int __esbmc_error_state = 0;
static int __esbmc_stream_mode = 0; // 0=read-only, 1=write-only, 2=read-write
static int __esbmc_currently_reading = 0; // 0=not reading, 1=currently reading
static int __esbmc_errno = 0;
static int __esbmc_pipe_fds[2] = {-1, -1}; // track pipe file descriptors

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

FILE *fopen(const char *filename, const char *m)
{
__ESBMC_HIDE:;
#if __ESBMC_SVCOMP
  FILE *f = (void *)1;
#else
  FILE *f = malloc(sizeof(FILE));
#endif
  return f;
}

int fclose(FILE *stream)
{
__ESBMC_HIDE:;
// Don't clear pending state here - let it remain for verification
#if __ESBMC_SVCOMP
#else
  free(stream);
#endif
  return nondet_int();
}

FILE *fdopen(int handle, const char *m)
{
__ESBMC_HIDE:;
  FILE *f = malloc(sizeof(FILE));
  return f;
}

// fgets reads a line from the specified stream and stores it
// into the string pointed to by str. It stops when either (n-1)
// characters are read, the newline character is read, or the
// end-of-file is reached, whichever comes first.
// source: https://www.cplusplus.com/reference/cstdio/fgets/
char *fgets(char *str, int size, FILE *stream)
{
__ESBMC_HIDE:;
  // Set currently reading state
  __esbmc_currently_reading = 1;

  // (rest of existing fgets implementation...)
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
  // just return nondet
  return nondet_int();
}

int ferror(FILE *stream)
{
__ESBMC_HIDE:;
  return __esbmc_error_state;
}

int fileno(FILE *stream)
{
__ESBMC_HIDE:;
  // just return nondet
  return nondet_int();
}

int fputs(const char *s, FILE *stream)
{
__ESBMC_HIDE:;
  // Clear reading state when writing
  __esbmc_currently_reading = 0;

  // Check if stream is writable (mode 1 or 2)
  if (__esbmc_stream_mode == 0)
  {
    __esbmc_error_state = nondet_int();
    __ESBMC_assume(__esbmc_error_state != 0);
    return EOF;
  }
  else
  {
    if (stream != NULL)
    {
      __esbmc_buffer_pending = nondet_uint();
      __ESBMC_assume(__esbmc_buffer_pending > 0);
    }
    int ret = nondet_int();
    __ESBMC_assume(ret >= 0);
    return ret;
  }
}

int fflush(FILE *stream)
{
__ESBMC_HIDE:;
  // Clear pending data when flushing
  __esbmc_buffer_pending = 0;
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
  // For valid file descriptors, succeed with positive bytes read
  if (fildes >= 0 && buf != NULL && nbyte > 0)
  {
    ssize_t nread = nondet_long();
    __ESBMC_assume(nread > 0 && nread <= nbyte);

    size_t i;
    for (i = 0; i < nread; i++)
      ((char *)buf)[i] = nondet_char();

    return nread;
  }
  else
  {
    return -1; // error case for invalid parameters
  }
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
  __esbmc_currently_reading = 0; // Reset reading state
}

size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
__ESBMC_HIDE:;
  size_t nwrite;
  __ESBMC_assume(nwrite <= nitems);
  return nwrite;
}

int fputs_unlocked(const char *s, FILE *stream)
{
__ESBMC_HIDE:;
  // Always set pending data when fputs_unlocked is called with valid parameters
  if (stream != NULL)
  {
    __esbmc_buffer_pending = nondet_uint();
    __ESBMC_assume(__esbmc_buffer_pending > 0);
  }
  return nondet_int();
}

FILE *fmemopen(void *buf, size_t size, const char *mode)
{
__ESBMC_HIDE:;
  __esbmc_buffer_pending = 0; // Initialize to empty buffer
  __esbmc_error_state = 0;    // Initialize to no error

  // Parse mode to determine stream capabilities and initial reading state
  if (mode != NULL)
  {
    if (mode[0] == 'r' && mode[1] == '\0')
    {
      __esbmc_stream_mode = 0; // read-only
      __esbmc_currently_reading =
        1; // read-only streams are always in read mode
    }
    else if (mode[0] == 'w')
    {
      __esbmc_stream_mode =
        (mode[1] == '+') ? 2 : 1;    // write-only or read-write
      __esbmc_currently_reading = 0; // write streams start not reading
    }
    else if (mode[0] == 'r' && mode[1] == '+')
    {
      __esbmc_stream_mode = 2;       // read-write
      __esbmc_currently_reading = 0; // read-write streams start not reading
    }
    else
    {
      __esbmc_stream_mode = 2; // default to read-write for other modes
      __esbmc_currently_reading = 0;
    }
  }
  else
  {
    __esbmc_stream_mode = 2; // default to read-write if mode is NULL
    __esbmc_currently_reading = 0;
  }

  FILE *f = malloc(sizeof(FILE));
  return f;
}

size_t __fpending(FILE *stream)
{
__ESBMC_HIDE:;
  return __esbmc_buffer_pending;
}

int ferror_unlocked(FILE *stream)
{
__ESBMC_HIDE:;
  return __esbmc_error_state;
}

int fputc(int c, FILE *stream)
{
__ESBMC_HIDE:;
  // Check if stream is writable (mode 1 or 2)
  if (__esbmc_stream_mode == 0)
  {
    // Read-only stream - always fail
    __esbmc_error_state = nondet_int();
    __ESBMC_assume(__esbmc_error_state != 0);
    return EOF;
  }
  else
  {
    // Writable stream - succeed
    return nondet_int();
  }
}

int __freading(FILE *stream)
{
__ESBMC_HIDE:;
  return __esbmc_currently_reading;
}

off_t lseek(int fd, off_t offset, int whence)
{
__ESBMC_HIDE:;
  // Check if this is a pipe file descriptor
  if (fd == __esbmc_pipe_fds[0] || fd == __esbmc_pipe_fds[1])
  {
    __esbmc_errno = 29; // ESPIPE
    return -1;
  }

  // For regular valid file descriptors
  if (fd >= 0)
  {
    if (whence == SEEK_SET)
    {
      if (offset < 0)
      {
        __esbmc_errno = 22; // EINVAL - Invalid argument
        return -1;
      }
      return offset; // return the requested offset for valid SEEK_SET
    }
    else
    {
      off_t new_offset = nondet_long();
      __ESBMC_assume(new_offset >= 0);
      return new_offset;
    }
  }
  else
    return -1;
}

int open(const char *pathname, int flags, ...)
{
__ESBMC_HIDE:;
  int fd = nondet_int();
  __ESBMC_assume(fd >= 0); // open returns non-negative fd on success
  return fd;
}

ssize_t write(int fd, const void *buf, size_t count)
{
__ESBMC_HIDE:;
  // For valid file descriptors, always succeed
  if (fd >= 0 && buf != NULL)
    return count; // success case - write all requested bytes
  else
    return -1; // error case for invalid parameters
}

int close(int fd)
{
__ESBMC_HIDE:;
  return nondet_int(); // can succeed (0) or fail (-1)
}

int *__errno_location(void)
{
__ESBMC_HIDE:;
  return &__esbmc_errno;
}

int pipe(int pipefd[2])
{
__ESBMC_HIDE:;
  // Always succeed for valid input
  if (pipefd != NULL)
  {
    // Success - assign positive file descriptors for pipes
    pipefd[0] = nondet_int();
    pipefd[1] = nondet_int();
    __ESBMC_assume(pipefd[0] > 2 && pipefd[1] > 2);
    __ESBMC_assume(pipefd[0] != pipefd[1]);
    __esbmc_pipe_fds[0] = pipefd[0];
    __esbmc_pipe_fds[1] = pipefd[1];

    return 0;
  }
  else
  {
    __esbmc_errno = nondet_int();
    return -1;
  }
}