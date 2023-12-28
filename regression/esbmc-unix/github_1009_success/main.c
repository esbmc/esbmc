// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2022 NIST
// SPDX-FileCopyrightText: 2022 The SV-Benchmarks Community
//
// SPDX-License-Identifier: CC0-1.0

typedef long unsigned int size_t;
typedef __builtin_va_list __gnuc_va_list;
typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct
{
  int __val[2];
} __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef long int __suseconds64_t;
typedef int __daddr_t;
typedef int __key_t;
typedef int __clockid_t;
typedef void *__timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __fsword_t;
typedef long int __ssize_t;
typedef long int __syscall_slong_t;
typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef char *__caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef int __sig_atomic_t;
typedef struct
{
  int __count;
  union
  {
    unsigned int __wch;
    char __wchb[4];
  } __value;
} __mbstate_t;
typedef struct _G_fpos_t
{
  __off_t __pos;
  __mbstate_t __state;
} __fpos_t;
typedef struct _G_fpos64_t
{
  __off64_t __pos;
  __mbstate_t __state;
} __fpos64_t;
struct _IO_FILE;
typedef struct _IO_FILE __FILE;
struct _IO_FILE;
typedef struct _IO_FILE FILE;
struct _IO_FILE;
struct _IO_marker;
struct _IO_codecvt;
struct _IO_wide_data;
typedef void _IO_lock_t;
struct _IO_FILE
{
  int _flags;
  char *_IO_read_ptr;
  char *_IO_read_end;
  char *_IO_read_base;
  char *_IO_write_base;
  char *_IO_write_ptr;
  char *_IO_write_end;
  char *_IO_buf_base;
  char *_IO_buf_end;
  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;
  struct _IO_marker *_markers;
  struct _IO_FILE *_chain;
  int _fileno;
  int _flags2;
  __off_t _old_offset;
  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];
  _IO_lock_t *_lock;
  __off64_t _offset;
  struct _IO_codecvt *_codecvt;
  struct _IO_wide_data *_wide_data;
  struct _IO_FILE *_freeres_list;
  void *_freeres_buf;
  size_t __pad5;
  int _mode;
  char _unused2[15 * sizeof(int) - 4 * sizeof(void *) - sizeof(size_t)];
};
typedef __gnuc_va_list va_list;
typedef __off_t off_t;
typedef __ssize_t ssize_t;
typedef __fpos_t fpos_t;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
extern int remove(const char *__filename)
  __attribute__((__nothrow__, __leaf__));
extern int rename(const char *__old, const char *__new)
  __attribute__((__nothrow__, __leaf__));
extern int
renameat(int __oldfd, const char *__old, int __newfd, const char *__new)
  __attribute__((__nothrow__, __leaf__));
extern int fclose(FILE *__stream);
extern FILE *tmpfile(void) __attribute__((__malloc__));
extern char *tmpnam(char[20]) __attribute__((__nothrow__, __leaf__));
extern char *tmpnam_r(char __s[20]) __attribute__((__nothrow__, __leaf__));
extern char *tempnam(const char *__dir, const char *__pfx)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern int fflush(FILE *__stream);
extern int fflush_unlocked(FILE *__stream);
extern FILE *
fopen(const char *__restrict __filename, const char *__restrict __modes)
  __attribute__((__malloc__));
extern FILE *freopen(
  const char *__restrict __filename,
  const char *__restrict __modes,
  FILE *__restrict __stream);
extern FILE *fdopen(int __fd, const char *__modes)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern FILE *fmemopen(void *__s, size_t __len, const char *__modes)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern FILE *open_memstream(char **__bufloc, size_t *__sizeloc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern void setbuf(FILE *__restrict __stream, char *__restrict __buf)
  __attribute__((__nothrow__, __leaf__));
extern int setvbuf(
  FILE *__restrict __stream,
  char *__restrict __buf,
  int __modes,
  size_t __n) __attribute__((__nothrow__, __leaf__));
extern void
setbuffer(FILE *__restrict __stream, char *__restrict __buf, size_t __size)
  __attribute__((__nothrow__, __leaf__));
extern void setlinebuf(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int
fprintf(FILE *__restrict __stream, const char *__restrict __format, ...);
extern int printf(const char *__restrict __format, ...);
extern int sprintf(char *__restrict __s, const char *__restrict __format, ...)
  __attribute__((__nothrow__));
extern int vfprintf(
  FILE *__restrict __s,
  const char *__restrict __format,
  __gnuc_va_list __arg);
extern int vprintf(const char *__restrict __format, __gnuc_va_list __arg);
extern int vsprintf(
  char *__restrict __s,
  const char *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__nothrow__));
extern int snprintf(
  char *__restrict __s,
  size_t __maxlen,
  const char *__restrict __format,
  ...) __attribute__((__nothrow__))
__attribute__((__format__(__printf__, 3, 4)));
extern int vsnprintf(
  char *__restrict __s,
  size_t __maxlen,
  const char *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__nothrow__))
__attribute__((__format__(__printf__, 3, 0)));
extern int
vdprintf(int __fd, const char *__restrict __fmt, __gnuc_va_list __arg)
  __attribute__((__format__(__printf__, 2, 0)));
extern int dprintf(int __fd, const char *__restrict __fmt, ...)
  __attribute__((__format__(__printf__, 2, 3)));
extern int
fscanf(FILE *__restrict __stream, const char *__restrict __format, ...);
extern int scanf(const char *__restrict __format, ...);
extern int
sscanf(const char *__restrict __s, const char *__restrict __format, ...)
  __attribute__((__nothrow__, __leaf__));
extern int
fscanf(FILE *__restrict __stream, const char *__restrict __format, ...) __asm__(
  ""
  "__isoc99_fscanf");
extern int scanf(const char *__restrict __format, ...) __asm__(
  ""
  "__isoc99_scanf");
extern int
sscanf(const char *__restrict __s, const char *__restrict __format, ...) __asm__(
  ""
  "__isoc99_sscanf") __attribute__((__nothrow__, __leaf__));
extern int vfscanf(
  FILE *__restrict __s,
  const char *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__format__(__scanf__, 2, 0)));
extern int vscanf(const char *__restrict __format, __gnuc_va_list __arg)
  __attribute__((__format__(__scanf__, 1, 0)));
extern int vsscanf(
  const char *__restrict __s,
  const char *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__nothrow__, __leaf__))
__attribute__((__format__(__scanf__, 2, 0)));
extern int
vfscanf(FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vfscanf") __attribute__((__format__(__scanf__, 2, 0)));
extern int
vscanf(const char *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vscanf") __attribute__((__format__(__scanf__, 1, 0)));
extern int
vsscanf(const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vsscanf") __attribute__((__nothrow__, __leaf__))
__attribute__((__format__(__scanf__, 2, 0)));
extern int fgetc(FILE *__stream);
extern int getc(FILE *__stream);
extern int getchar(void);
extern int getc_unlocked(FILE *__stream);
extern int getchar_unlocked(void);
extern int fgetc_unlocked(FILE *__stream);
extern int fputc(int __c, FILE *__stream);
extern int putc(int __c, FILE *__stream);
extern int putchar(int __c);
extern int fputc_unlocked(int __c, FILE *__stream);
extern int putc_unlocked(int __c, FILE *__stream);
extern int putchar_unlocked(int __c);
extern int getw(FILE *__stream);
extern int putw(int __w, FILE *__stream);
extern char *fgets(char *__restrict __s, int __n, FILE *__restrict __stream);
extern __ssize_t __getdelim(
  char **__restrict __lineptr,
  size_t *__restrict __n,
  int __delimiter,
  FILE *__restrict __stream);
extern __ssize_t getdelim(
  char **__restrict __lineptr,
  size_t *__restrict __n,
  int __delimiter,
  FILE *__restrict __stream);
extern __ssize_t getline(
  char **__restrict __lineptr,
  size_t *__restrict __n,
  FILE *__restrict __stream);
extern int fputs(const char *__restrict __s, FILE *__restrict __stream);
extern int puts(const char *__s);
extern int ungetc(int __c, FILE *__stream);
extern size_t fread(
  void *__restrict __ptr,
  size_t __size,
  size_t __n,
  FILE *__restrict __stream);
extern size_t fwrite(
  const void *__restrict __ptr,
  size_t __size,
  size_t __n,
  FILE *__restrict __s);
extern size_t fread_unlocked(
  void *__restrict __ptr,
  size_t __size,
  size_t __n,
  FILE *__restrict __stream);
extern size_t fwrite_unlocked(
  const void *__restrict __ptr,
  size_t __size,
  size_t __n,
  FILE *__restrict __stream);
extern int fseek(FILE *__stream, long int __off, int __whence);
extern long int ftell(FILE *__stream);
extern void rewind(FILE *__stream);
extern int fseeko(FILE *__stream, __off_t __off, int __whence);
extern __off_t ftello(FILE *__stream);
extern int fgetpos(FILE *__restrict __stream, fpos_t *__restrict __pos);
extern int fsetpos(FILE *__stream, const fpos_t *__pos);
extern void clearerr(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int feof(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int ferror(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern void clearerr_unlocked(FILE *__stream)
  __attribute__((__nothrow__, __leaf__));
extern int feof_unlocked(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int ferror_unlocked(FILE *__stream)
  __attribute__((__nothrow__, __leaf__));
extern void perror(const char *__s);
extern int fileno(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int fileno_unlocked(FILE *__stream)
  __attribute__((__nothrow__, __leaf__));
extern int pclose(FILE *__stream);
extern FILE *popen(const char *__command, const char *__modes)
  __attribute__((__malloc__));
extern char *ctermid(char *__s) __attribute__((__nothrow__, __leaf__));
extern void flockfile(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int ftrylockfile(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern void funlockfile(FILE *__stream) __attribute__((__nothrow__, __leaf__));
extern int __uflow(FILE *);
extern int __overflow(FILE *, int);

typedef int wchar_t;

typedef enum
{
  P_ALL,
  P_PID,
  P_PGID
} idtype_t;
typedef struct
{
  int quot;
  int rem;
} div_t;
typedef struct
{
  long int quot;
  long int rem;
} ldiv_t;
__extension__ typedef struct
{
  long long int quot;
  long long int rem;
} lldiv_t;
extern size_t __ctype_get_mb_cur_max(void)
  __attribute__((__nothrow__, __leaf__));
extern double atof(const char *__nptr) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)));
extern int atoi(const char *__nptr) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)));
extern long int atol(const char *__nptr) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)));
__extension__ extern long long int atoll(const char *__nptr)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern double strtod(const char *__restrict __nptr, char **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern float strtof(const char *__restrict __nptr, char **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern long double
strtold(const char *__restrict __nptr, char **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern long int
strtol(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern unsigned long int
strtoul(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
__extension__ extern long long int
strtoq(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
__extension__ extern unsigned long long int
strtouq(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
__extension__ extern long long int
strtoll(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
__extension__ extern unsigned long long int
strtoull(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern char *l64a(long int __n) __attribute__((__nothrow__, __leaf__));
extern long int a64l(const char *__s) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)));

typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;
typedef __loff_t loff_t;
typedef __ino_t ino_t;
typedef __dev_t dev_t;
typedef __gid_t gid_t;
typedef __mode_t mode_t;
typedef __nlink_t nlink_t;
typedef __uid_t uid_t;
typedef __pid_t pid_t;
typedef __id_t id_t;
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;
typedef __key_t key_t;
typedef __clock_t clock_t;
typedef __clockid_t clockid_t;
typedef __time_t time_t;
typedef __timer_t timer_t;
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef __uint8_t u_int8_t;
typedef __uint16_t u_int16_t;
typedef __uint32_t u_int32_t;
typedef __uint64_t u_int64_t;
typedef int register_t __attribute__((__mode__(__word__)));
static __inline __uint16_t __bswap_16(__uint16_t __bsx)
{
  return __builtin_bswap16(__bsx);
}
static __inline __uint32_t __bswap_32(__uint32_t __bsx)
{
  return __builtin_bswap32(__bsx);
}
__extension__ static __inline __uint64_t __bswap_64(__uint64_t __bsx)
{
  return __builtin_bswap64(__bsx);
}
static __inline __uint16_t __uint16_identity(__uint16_t __x)
{
  return __x;
}
static __inline __uint32_t __uint32_identity(__uint32_t __x)
{
  return __x;
}
static __inline __uint64_t __uint64_identity(__uint64_t __x)
{
  return __x;
}
typedef struct
{
  unsigned long int __val[(1024 / (8 * sizeof(unsigned long int)))];
} __sigset_t;
typedef __sigset_t sigset_t;
struct timeval
{
  __time_t tv_sec;
  __suseconds_t tv_usec;
};
struct timespec
{
  __time_t tv_sec;
  __syscall_slong_t tv_nsec;
};
typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
typedef struct
{
  __fd_mask __fds_bits[1024 / (8 * (int)sizeof(__fd_mask))];
} fd_set;
typedef __fd_mask fd_mask;

extern int select(
  int __nfds,
  fd_set *__restrict __readfds,
  fd_set *__restrict __writefds,
  fd_set *__restrict __exceptfds,
  struct timeval *__restrict __timeout);
extern int pselect(
  int __nfds,
  fd_set *__restrict __readfds,
  fd_set *__restrict __writefds,
  fd_set *__restrict __exceptfds,
  const struct timespec *__restrict __timeout,
  const __sigset_t *__restrict __sigmask);

typedef __blksize_t blksize_t;
typedef __blkcnt_t blkcnt_t;
typedef __fsblkcnt_t fsblkcnt_t;
typedef __fsfilcnt_t fsfilcnt_t;
typedef union
{
  __extension__ unsigned long long int __value64;
  struct
  {
    unsigned int __low;
    unsigned int __high;
  } __value32;
} __atomic_wide_counter;
typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
typedef struct __pthread_internal_slist
{
  struct __pthread_internal_slist *__next;
} __pthread_slist_t;
struct __pthread_mutex_s
{
  int __lock;
  unsigned int __count;
  int __owner;
  unsigned int __nusers;
  int __kind;
  short __spins;
  short __elision;
  __pthread_list_t __list;
};
struct __pthread_rwlock_arch_t
{
  unsigned int __readers;
  unsigned int __writers;
  unsigned int __wrphase_futex;
  unsigned int __writers_futex;
  unsigned int __pad3;
  unsigned int __pad4;
  int __cur_writer;
  int __shared;
  signed char __rwelision;
  unsigned char __pad1[7];
  unsigned long int __pad2;
  unsigned int __flags;
};
struct __pthread_cond_s
{
  __atomic_wide_counter __wseq;
  __atomic_wide_counter __g1_start;
  unsigned int __g_refs[2];
  unsigned int __g_size[2];
  unsigned int __g1_orig_size;
  unsigned int __wrefs;
  unsigned int __g_signals[2];
};
typedef unsigned int __tss_t;
typedef unsigned long int __thrd_t;
typedef struct
{
  int __data;
} __once_flag;
typedef unsigned long int pthread_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_mutexattr_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_condattr_t;
typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
union pthread_attr_t
{
  char __size[56];
  long int __align;
};
typedef union pthread_attr_t pthread_attr_t;
typedef union
{
  struct __pthread_mutex_s __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;
typedef union
{
  struct __pthread_cond_s __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;
typedef union
{
  struct __pthread_rwlock_arch_t __data;
  char __size[56];
  long int __align;
} pthread_rwlock_t;
typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;
typedef volatile int pthread_spinlock_t;
typedef union
{
  char __size[32];
  long int __align;
} pthread_barrier_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;

extern long int random(void) __attribute__((__nothrow__, __leaf__));
extern void srandom(unsigned int __seed) __attribute__((__nothrow__, __leaf__));
extern char *initstate(unsigned int __seed, char *__statebuf, size_t __statelen)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern char *setstate(char *__statebuf) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
struct random_data
{
  int32_t *fptr;
  int32_t *rptr;
  int32_t *state;
  int rand_type;
  int rand_deg;
  int rand_sep;
  int32_t *end_ptr;
};
extern int
random_r(struct random_data *__restrict __buf, int32_t *__restrict __result)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int srandom_r(unsigned int __seed, struct random_data *__buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int initstate_r(
  unsigned int __seed,
  char *__restrict __statebuf,
  size_t __statelen,
  struct random_data *__restrict __buf) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2, 4)));
extern int
setstate_r(char *__restrict __statebuf, struct random_data *__restrict __buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int rand(void) __attribute__((__nothrow__, __leaf__));
extern void srand(unsigned int __seed) __attribute__((__nothrow__, __leaf__));
extern int rand_r(unsigned int *__seed) __attribute__((__nothrow__, __leaf__));
extern double drand48(void) __attribute__((__nothrow__, __leaf__));
extern double erand48(unsigned short int __xsubi[3])
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern long int lrand48(void) __attribute__((__nothrow__, __leaf__));
extern long int nrand48(unsigned short int __xsubi[3])
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern long int mrand48(void) __attribute__((__nothrow__, __leaf__));
extern long int jrand48(unsigned short int __xsubi[3])
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern void srand48(long int __seedval) __attribute__((__nothrow__, __leaf__));
extern unsigned short int *seed48(unsigned short int __seed16v[3])
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern void lcong48(unsigned short int __param[7])
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
struct drand48_data
{
  unsigned short int __x[3];
  unsigned short int __old_x[3];
  unsigned short int __c;
  unsigned short int __init;
  __extension__ unsigned long long int __a;
};
extern int
drand48_r(struct drand48_data *__restrict __buffer, double *__restrict __result)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int erand48_r(
  unsigned short int __xsubi[3],
  struct drand48_data *__restrict __buffer,
  double *__restrict __result) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern int lrand48_r(
  struct drand48_data *__restrict __buffer,
  long int *__restrict __result) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern int nrand48_r(
  unsigned short int __xsubi[3],
  struct drand48_data *__restrict __buffer,
  long int *__restrict __result) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern int mrand48_r(
  struct drand48_data *__restrict __buffer,
  long int *__restrict __result) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern int jrand48_r(
  unsigned short int __xsubi[3],
  struct drand48_data *__restrict __buffer,
  long int *__restrict __result) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern int srand48_r(long int __seedval, struct drand48_data *__buffer)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int
seed48_r(unsigned short int __seed16v[3], struct drand48_data *__buffer)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int
lcong48_r(unsigned short int __param[7], struct drand48_data *__buffer)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern void *malloc(size_t __size) __attribute__((__nothrow__, __leaf__))
__attribute__((__malloc__)) __attribute__((__alloc_size__(1)));
extern void *calloc(size_t __nmemb, size_t __size)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__))
  __attribute__((__alloc_size__(1, 2)));
extern void *realloc(void *__ptr, size_t __size)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__warn_unused_result__))
  __attribute__((__alloc_size__(2)));
extern void free(void *__ptr) __attribute__((__nothrow__, __leaf__));
extern void *reallocarray(void *__ptr, size_t __nmemb, size_t __size)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__warn_unused_result__))
  __attribute__((__alloc_size__(2, 3)));
extern void *reallocarray(void *__ptr, size_t __nmemb, size_t __size)
  __attribute__((__nothrow__, __leaf__));

extern void *alloca(size_t __size) __attribute__((__nothrow__, __leaf__));

extern void *valloc(size_t __size) __attribute__((__nothrow__, __leaf__))
__attribute__((__malloc__)) __attribute__((__alloc_size__(1)));
extern int posix_memalign(void **__memptr, size_t __alignment, size_t __size)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern void *aligned_alloc(size_t __alignment, size_t __size)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__))
  __attribute__((__alloc_align__(1))) __attribute__((__alloc_size__(2)));
extern void abort(void) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
extern int atexit(void (*__func)(void)) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int at_quick_exit(void (*__func)(void))
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int on_exit(void (*__func)(int __status, void *__arg), void *__arg)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern void exit(int __status) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
extern void quick_exit(int __status) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
extern void _Exit(int __status) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
extern char *getenv(const char *__name) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int putenv(char *__string) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int setenv(const char *__name, const char *__value, int __replace)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int unsetenv(const char *__name) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int clearenv(void) __attribute__((__nothrow__, __leaf__));
extern char *mktemp(char *__template) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int mkstemp(char *__template) __attribute__((__nonnull__(1)));
extern int mkstemps(char *__template, int __suffixlen)
  __attribute__((__nonnull__(1)));
extern char *mkdtemp(char *__template) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern int system(const char *__command);
extern char *
realpath(const char *__restrict __name, char *__restrict __resolved)
  __attribute__((__nothrow__, __leaf__));
typedef int (*__compar_fn_t)(const void *, const void *);
extern void *bsearch(
  const void *__key,
  const void *__base,
  size_t __nmemb,
  size_t __size,
  __compar_fn_t __compar) __attribute__((__nonnull__(1, 2, 5)));
extern void
qsort(void *__base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
  __attribute__((__nonnull__(1, 4)));
extern int abs(int __x) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
extern long int labs(long int __x) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
__extension__ extern long long int llabs(long long int __x)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern div_t div(int __numer, int __denom)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern ldiv_t ldiv(long int __numer, long int __denom)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
__extension__ extern lldiv_t lldiv(long long int __numer, long long int __denom)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern char *ecvt(
  double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4)));
extern char *fcvt(
  double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4)));
extern char *gcvt(double __value, int __ndigit, char *__buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(3)));
extern char *qecvt(
  long double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4)));
extern char *qfcvt(
  long double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4)));
extern char *qgcvt(long double __value, int __ndigit, char *__buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(3)));
extern int ecvt_r(
  double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign,
  char *__restrict __buf,
  size_t __len) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4, 5)));
extern int fcvt_r(
  double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign,
  char *__restrict __buf,
  size_t __len) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4, 5)));
extern int qecvt_r(
  long double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign,
  char *__restrict __buf,
  size_t __len) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4, 5)));
extern int qfcvt_r(
  long double __value,
  int __ndigit,
  int *__restrict __decpt,
  int *__restrict __sign,
  char *__restrict __buf,
  size_t __len) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(3, 4, 5)));
extern int mblen(const char *__s, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern int
mbtowc(wchar_t *__restrict __pwc, const char *__restrict __s, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern int wctomb(char *__s, wchar_t __wchar)
  __attribute__((__nothrow__, __leaf__));
extern size_t
mbstowcs(wchar_t *__restrict __pwcs, const char *__restrict __s, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern size_t
wcstombs(char *__restrict __s, const wchar_t *__restrict __pwcs, size_t __n)
  __attribute__((__nothrow__, __leaf__))

  ;
extern int rpmatch(const char *__response)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int getsubopt(
  char **__restrict __optionp,
  char *const *__restrict __tokens,
  char **__restrict __valuep) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2, 3)));
extern int getloadavg(double __loadavg[], int __nelem)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));

typedef long int ptrdiff_t;
typedef struct
{
  long long __max_align_ll __attribute__((__aligned__(__alignof__(long long))));
  long double __max_align_ld
    __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;
struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
  long int tm_gmtoff;
  const char *tm_zone;
};
struct itimerspec
{
  struct timespec it_interval;
  struct timespec it_value;
};
struct sigevent;
struct __locale_struct
{
  struct __locale_data *__locales[13];
  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;
  const char *__names[13];
};
typedef struct __locale_struct *__locale_t;
typedef __locale_t locale_t;

extern clock_t clock(void) __attribute__((__nothrow__, __leaf__));
extern time_t time(time_t *__timer) __attribute__((__nothrow__, __leaf__));
extern double difftime(time_t __time1, time_t __time0)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern time_t mktime(struct tm *__tp) __attribute__((__nothrow__, __leaf__));
extern size_t strftime(
  char *__restrict __s,
  size_t __maxsize,
  const char *__restrict __format,
  const struct tm *__restrict __tp) __attribute__((__nothrow__, __leaf__));
extern size_t strftime_l(
  char *__restrict __s,
  size_t __maxsize,
  const char *__restrict __format,
  const struct tm *__restrict __tp,
  locale_t __loc) __attribute__((__nothrow__, __leaf__));
extern struct tm *gmtime(const time_t *__timer)
  __attribute__((__nothrow__, __leaf__));
extern struct tm *localtime(const time_t *__timer)
  __attribute__((__nothrow__, __leaf__));
extern struct tm *
gmtime_r(const time_t *__restrict __timer, struct tm *__restrict __tp)
  __attribute__((__nothrow__, __leaf__));
extern struct tm *
localtime_r(const time_t *__restrict __timer, struct tm *__restrict __tp)
  __attribute__((__nothrow__, __leaf__));
extern char *asctime(const struct tm *__tp)
  __attribute__((__nothrow__, __leaf__));
extern char *ctime(const time_t *__timer)
  __attribute__((__nothrow__, __leaf__));
extern char *asctime_r(const struct tm *__restrict __tp, char *__restrict __buf)
  __attribute__((__nothrow__, __leaf__));
extern char *ctime_r(const time_t *__restrict __timer, char *__restrict __buf)
  __attribute__((__nothrow__, __leaf__));
extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;
extern char *tzname[2];
extern void tzset(void) __attribute__((__nothrow__, __leaf__));
extern int daylight;
extern long int timezone;
extern time_t timegm(struct tm *__tp) __attribute__((__nothrow__, __leaf__));
extern time_t timelocal(struct tm *__tp) __attribute__((__nothrow__, __leaf__));
extern int dysize(int __year) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
extern int nanosleep(
  const struct timespec *__requested_time,
  struct timespec *__remaining);
extern int clock_getres(clockid_t __clock_id, struct timespec *__res)
  __attribute__((__nothrow__, __leaf__));
extern int clock_gettime(clockid_t __clock_id, struct timespec *__tp)
  __attribute__((__nothrow__, __leaf__));
extern int clock_settime(clockid_t __clock_id, const struct timespec *__tp)
  __attribute__((__nothrow__, __leaf__));
extern int clock_nanosleep(
  clockid_t __clock_id,
  int __flags,
  const struct timespec *__req,
  struct timespec *__rem);
extern int clock_getcpuclockid(pid_t __pid, clockid_t *__clock_id)
  __attribute__((__nothrow__, __leaf__));
extern int timer_create(
  clockid_t __clock_id,
  struct sigevent *__restrict __evp,
  timer_t *__restrict __timerid) __attribute__((__nothrow__, __leaf__));
extern int timer_delete(timer_t __timerid)
  __attribute__((__nothrow__, __leaf__));
extern int timer_settime(
  timer_t __timerid,
  int __flags,
  const struct itimerspec *__restrict __value,
  struct itimerspec *__restrict __ovalue)
  __attribute__((__nothrow__, __leaf__));
extern int timer_gettime(timer_t __timerid, struct itimerspec *__value)
  __attribute__((__nothrow__, __leaf__));
extern int timer_getoverrun(timer_t __timerid)
  __attribute__((__nothrow__, __leaf__));
extern int timespec_get(struct timespec *__ts, int __base)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));

extern void *
memcpy(void *__restrict __dest, const void *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern void *memmove(void *__dest, const void *__src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern void *memccpy(
  void *__restrict __dest,
  const void *__restrict __src,
  int __c,
  size_t __n) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1, 2)));
extern void *memset(void *__s, int __c, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int memcmp(const void *__s1, const void *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int __memcmpeq(const void *__s1, const void *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern void *memchr(const void *__s, int __c, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern char *strcpy(char *__restrict __dest, const char *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *
strncpy(char *__restrict __dest, const char *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *strcat(char *__restrict __dest, const char *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *
strncat(char *__restrict __dest, const char *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int strcmp(const char *__s1, const char *__s2)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int strncmp(const char *__s1, const char *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int strcoll(const char *__s1, const char *__s2)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern size_t
strxfrm(char *__restrict __dest, const char *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int strcoll_l(const char *__s1, const char *__s2, locale_t __l)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2, 3)));
extern size_t
strxfrm_l(char *__dest, const char *__src, size_t __n, locale_t __l)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2, 4)));
extern char *strdup(const char *__s) __attribute__((__nothrow__, __leaf__))
__attribute__((__malloc__)) __attribute__((__nonnull__(1)));
extern char *strndup(const char *__string, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__))
  __attribute__((__nonnull__(1)));
extern char *strchr(const char *__s, int __c)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern char *strrchr(const char *__s, int __c)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern size_t strcspn(const char *__s, const char *__reject)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern size_t strspn(const char *__s, const char *__accept)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern char *strpbrk(const char *__s, const char *__accept)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern char *strstr(const char *__haystack, const char *__needle)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern char *strtok(char *__restrict __s, const char *__restrict __delim)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern char *__strtok_r(
  char *__restrict __s,
  const char *__restrict __delim,
  char **__restrict __save_ptr) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2, 3)));
extern char *strtok_r(
  char *__restrict __s,
  const char *__restrict __delim,
  char **__restrict __save_ptr) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2, 3)));
extern size_t strlen(const char *__s) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)));
extern size_t strnlen(const char *__string, size_t __maxlen)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern char *strerror(int __errnum) __attribute__((__nothrow__, __leaf__));
extern int strerror_r(int __errnum, char *__buf, size_t __buflen) __asm__(
  ""
  "__xpg_strerror_r") __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2)));
extern char *strerror_l(int __errnum, locale_t __l)
  __attribute__((__nothrow__, __leaf__));

extern int bcmp(const void *__s1, const void *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern void bcopy(const void *__src, void *__dest, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern void bzero(void *__s, size_t __n) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(1)));
extern char *index(const char *__s, int __c)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern char *rindex(const char *__s, int __c)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1)));
extern int ffs(int __i) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
extern int ffsl(long int __l) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
__extension__ extern int ffsll(long long int __ll)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern int strcasecmp(const char *__s1, const char *__s2)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int strncasecmp(const char *__s1, const char *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int strcasecmp_l(const char *__s1, const char *__s2, locale_t __loc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2, 3)));
extern int
strncasecmp_l(const char *__s1, const char *__s2, size_t __n, locale_t __loc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2, 4)));

extern void explicit_bzero(void *__s, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern char *strsep(char **__restrict __stringp, const char *__restrict __delim)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *strsignal(int __sig) __attribute__((__nothrow__, __leaf__));
extern char *__stpcpy(char *__restrict __dest, const char *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *stpcpy(char *__restrict __dest, const char *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *
__stpncpy(char *__restrict __dest, const char *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern char *
stpncpy(char *__restrict __dest, const char *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));

typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
typedef __int_least8_t int_least8_t;
typedef __int_least16_t int_least16_t;
typedef __int_least32_t int_least32_t;
typedef __int_least64_t int_least64_t;
typedef __uint_least8_t uint_least8_t;
typedef __uint_least16_t uint_least16_t;
typedef __uint_least32_t uint_least32_t;
typedef __uint_least64_t uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef long int intptr_t;
typedef unsigned long int uintptr_t;
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;

enum
{
  _ISupper = ((0) < 8 ? ((1 << (0)) << 8) : ((1 << (0)) >> 8)),
  _ISlower = ((1) < 8 ? ((1 << (1)) << 8) : ((1 << (1)) >> 8)),
  _ISalpha = ((2) < 8 ? ((1 << (2)) << 8) : ((1 << (2)) >> 8)),
  _ISdigit = ((3) < 8 ? ((1 << (3)) << 8) : ((1 << (3)) >> 8)),
  _ISxdigit = ((4) < 8 ? ((1 << (4)) << 8) : ((1 << (4)) >> 8)),
  _ISspace = ((5) < 8 ? ((1 << (5)) << 8) : ((1 << (5)) >> 8)),
  _ISprint = ((6) < 8 ? ((1 << (6)) << 8) : ((1 << (6)) >> 8)),
  _ISgraph = ((7) < 8 ? ((1 << (7)) << 8) : ((1 << (7)) >> 8)),
  _ISblank = ((8) < 8 ? ((1 << (8)) << 8) : ((1 << (8)) >> 8)),
  _IScntrl = ((9) < 8 ? ((1 << (9)) << 8) : ((1 << (9)) >> 8)),
  _ISpunct = ((10) < 8 ? ((1 << (10)) << 8) : ((1 << (10)) >> 8)),
  _ISalnum = ((11) < 8 ? ((1 << (11)) << 8) : ((1 << (11)) >> 8))
};
extern const unsigned short int **__ctype_b_loc(void)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern const __int32_t **__ctype_tolower_loc(void)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern const __int32_t **__ctype_toupper_loc(void)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern int isalnum(int) __attribute__((__nothrow__, __leaf__));
extern int isalpha(int) __attribute__((__nothrow__, __leaf__));
extern int iscntrl(int) __attribute__((__nothrow__, __leaf__));
extern int isdigit(int) __attribute__((__nothrow__, __leaf__));
extern int islower(int) __attribute__((__nothrow__, __leaf__));
extern int isgraph(int) __attribute__((__nothrow__, __leaf__));
extern int isprint(int) __attribute__((__nothrow__, __leaf__));
extern int ispunct(int) __attribute__((__nothrow__, __leaf__));
extern int isspace(int) __attribute__((__nothrow__, __leaf__));
extern int isupper(int) __attribute__((__nothrow__, __leaf__));
extern int isxdigit(int) __attribute__((__nothrow__, __leaf__));
extern int tolower(int __c) __attribute__((__nothrow__, __leaf__));
extern int toupper(int __c) __attribute__((__nothrow__, __leaf__));
extern int isblank(int) __attribute__((__nothrow__, __leaf__));
extern int isascii(int __c) __attribute__((__nothrow__, __leaf__));
extern int toascii(int __c) __attribute__((__nothrow__, __leaf__));
extern int _toupper(int) __attribute__((__nothrow__, __leaf__));
extern int _tolower(int) __attribute__((__nothrow__, __leaf__));
extern int isalnum_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isalpha_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int iscntrl_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isdigit_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int islower_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isgraph_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isprint_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int ispunct_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isspace_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isupper_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isxdigit_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int isblank_l(int, locale_t) __attribute__((__nothrow__, __leaf__));
extern int __tolower_l(int __c, locale_t __l)
  __attribute__((__nothrow__, __leaf__));
extern int tolower_l(int __c, locale_t __l)
  __attribute__((__nothrow__, __leaf__));
extern int __toupper_l(int __c, locale_t __l)
  __attribute__((__nothrow__, __leaf__));
extern int toupper_l(int __c, locale_t __l)
  __attribute__((__nothrow__, __leaf__));

struct flock
{
  short int l_type;
  short int l_whence;
  __off_t l_start;
  __off_t l_len;
  __pid_t l_pid;
};

struct stat
{
  __dev_t st_dev;
  __ino_t st_ino;
  __nlink_t st_nlink;
  __mode_t st_mode;
  __uid_t st_uid;
  __gid_t st_gid;
  int __pad0;
  __dev_t st_rdev;
  __off_t st_size;
  __blksize_t st_blksize;
  __blkcnt_t st_blocks;
  struct timespec st_atim;
  struct timespec st_mtim;
  struct timespec st_ctim;
  __syscall_slong_t __glibc_reserved[3];
};
extern int fcntl(int __fd, int __cmd, ...);
extern int open(const char *__file, int __oflag, ...)
  __attribute__((__nonnull__(1)));
extern int openat(int __fd, const char *__file, int __oflag, ...)
  __attribute__((__nonnull__(2)));
extern int creat(const char *__file, mode_t __mode)
  __attribute__((__nonnull__(1)));
extern int lockf(int __fd, int __cmd, off_t __len);
extern int posix_fadvise(int __fd, off_t __offset, off_t __len, int __advise)
  __attribute__((__nothrow__, __leaf__));
extern int posix_fallocate(int __fd, off_t __offset, off_t __len);

extern int stat(const char *__restrict __file, struct stat *__restrict __buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int fstat(int __fd, struct stat *__buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int fstatat(
  int __fd,
  const char *__restrict __file,
  struct stat *__restrict __buf,
  int __flag) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2, 3)));
extern int lstat(const char *__restrict __file, struct stat *__restrict __buf)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int chmod(const char *__file, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int lchmod(const char *__file, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int fchmod(int __fd, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__));
extern int fchmodat(int __fd, const char *__file, __mode_t __mode, int __flag)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern __mode_t umask(__mode_t __mask) __attribute__((__nothrow__, __leaf__));
extern int mkdir(const char *__path, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int mkdirat(int __fd, const char *__path, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int mknod(const char *__path, __mode_t __mode, __dev_t __dev)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int mknodat(int __fd, const char *__path, __mode_t __mode, __dev_t __dev)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int mkfifo(const char *__path, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1)));
extern int mkfifoat(int __fd, const char *__path, __mode_t __mode)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(2)));
extern int utimensat(
  int __fd,
  const char *__path,
  const struct timespec __times[2],
  int __flags) __attribute__((__nothrow__, __leaf__))
__attribute__((__nonnull__(2)));
extern int futimens(int __fd, const struct timespec __times[2])
  __attribute__((__nothrow__, __leaf__));

typedef struct _twoIntsStruct
{
  int intOne;
  int intTwo;
} twoIntsStruct;
extern const int GLOBAL_CONST_TRUE;
extern const int GLOBAL_CONST_FALSE;
extern const int GLOBAL_CONST_FIVE;
extern int globalTrue;
extern int globalFalse;
extern int globalFive;
void printLine(const char *line);
void printWLine(const wchar_t *line);
void printIntLine(int intNumber);
void printShortLine(short shortNumber);
void printFloatLine(float floatNumber);
void printLongLine(long longNumber);
void printLongLongLine(int64_t longLongIntNumber);
void printSizeTLine(size_t sizeTNumber);
void printHexCharLine(char charHex);
void printWcharLine(wchar_t wideChar);
void printUnsignedLine(unsigned unsignedNumber);
void printHexUnsignedCharLine(unsigned char unsignedCharacter);
void printDoubleLine(double doubleNumber);
void printStructLine(const twoIntsStruct *structTwoIntsStruct);
void printBytesLine(const unsigned char *bytes, size_t numBytes);
size_t decodeHexChars(unsigned char *bytes, size_t numBytes, const char *hex);
size_t
decodeHexWChars(unsigned char *bytes, size_t numBytes, const wchar_t *hex);
int globalReturnsTrue();
int globalReturnsFalse();
int globalReturnsTrueOrFalse();
extern int globalArgc;
extern char **globalArgv;
typedef int __gwchar_t;

typedef struct
{
  long int quot;
  long int rem;
} imaxdiv_t;
extern intmax_t imaxabs(intmax_t __n) __attribute__((__nothrow__, __leaf__))
__attribute__((__const__));
extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__const__));
extern intmax_t
strtoimax(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__));
extern uintmax_t
strtoumax(const char *__restrict __nptr, char **__restrict __endptr, int __base)
  __attribute__((__nothrow__, __leaf__));
extern intmax_t wcstoimax(
  const __gwchar_t *__restrict __nptr,
  __gwchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));
extern uintmax_t wcstoumax(
  const __gwchar_t *__restrict __nptr,
  __gwchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));

typedef unsigned int wint_t;
typedef unsigned long int wctype_t;
enum
{
  __ISwupper = 0,
  __ISwlower = 1,
  __ISwalpha = 2,
  __ISwdigit = 3,
  __ISwxdigit = 4,
  __ISwspace = 5,
  __ISwprint = 6,
  __ISwgraph = 7,
  __ISwblank = 8,
  __ISwcntrl = 9,
  __ISwpunct = 10,
  __ISwalnum = 11,
  _ISwupper =
    ((__ISwupper) < 8
       ? (int)((1UL << (__ISwupper)) << 24)
       : ((__ISwupper) < 16
            ? (int)((1UL << (__ISwupper)) << 8)
            : ((__ISwupper) < 24 ? (int)((1UL << (__ISwupper)) >> 8)
                                 : (int)((1UL << (__ISwupper)) >> 24)))),
  _ISwlower =
    ((__ISwlower) < 8
       ? (int)((1UL << (__ISwlower)) << 24)
       : ((__ISwlower) < 16
            ? (int)((1UL << (__ISwlower)) << 8)
            : ((__ISwlower) < 24 ? (int)((1UL << (__ISwlower)) >> 8)
                                 : (int)((1UL << (__ISwlower)) >> 24)))),
  _ISwalpha =
    ((__ISwalpha) < 8
       ? (int)((1UL << (__ISwalpha)) << 24)
       : ((__ISwalpha) < 16
            ? (int)((1UL << (__ISwalpha)) << 8)
            : ((__ISwalpha) < 24 ? (int)((1UL << (__ISwalpha)) >> 8)
                                 : (int)((1UL << (__ISwalpha)) >> 24)))),
  _ISwdigit =
    ((__ISwdigit) < 8
       ? (int)((1UL << (__ISwdigit)) << 24)
       : ((__ISwdigit) < 16
            ? (int)((1UL << (__ISwdigit)) << 8)
            : ((__ISwdigit) < 24 ? (int)((1UL << (__ISwdigit)) >> 8)
                                 : (int)((1UL << (__ISwdigit)) >> 24)))),
  _ISwxdigit =
    ((__ISwxdigit) < 8
       ? (int)((1UL << (__ISwxdigit)) << 24)
       : ((__ISwxdigit) < 16
            ? (int)((1UL << (__ISwxdigit)) << 8)
            : ((__ISwxdigit) < 24 ? (int)((1UL << (__ISwxdigit)) >> 8)
                                  : (int)((1UL << (__ISwxdigit)) >> 24)))),
  _ISwspace =
    ((__ISwspace) < 8
       ? (int)((1UL << (__ISwspace)) << 24)
       : ((__ISwspace) < 16
            ? (int)((1UL << (__ISwspace)) << 8)
            : ((__ISwspace) < 24 ? (int)((1UL << (__ISwspace)) >> 8)
                                 : (int)((1UL << (__ISwspace)) >> 24)))),
  _ISwprint =
    ((__ISwprint) < 8
       ? (int)((1UL << (__ISwprint)) << 24)
       : ((__ISwprint) < 16
            ? (int)((1UL << (__ISwprint)) << 8)
            : ((__ISwprint) < 24 ? (int)((1UL << (__ISwprint)) >> 8)
                                 : (int)((1UL << (__ISwprint)) >> 24)))),
  _ISwgraph =
    ((__ISwgraph) < 8
       ? (int)((1UL << (__ISwgraph)) << 24)
       : ((__ISwgraph) < 16
            ? (int)((1UL << (__ISwgraph)) << 8)
            : ((__ISwgraph) < 24 ? (int)((1UL << (__ISwgraph)) >> 8)
                                 : (int)((1UL << (__ISwgraph)) >> 24)))),
  _ISwblank =
    ((__ISwblank) < 8
       ? (int)((1UL << (__ISwblank)) << 24)
       : ((__ISwblank) < 16
            ? (int)((1UL << (__ISwblank)) << 8)
            : ((__ISwblank) < 24 ? (int)((1UL << (__ISwblank)) >> 8)
                                 : (int)((1UL << (__ISwblank)) >> 24)))),
  _ISwcntrl =
    ((__ISwcntrl) < 8
       ? (int)((1UL << (__ISwcntrl)) << 24)
       : ((__ISwcntrl) < 16
            ? (int)((1UL << (__ISwcntrl)) << 8)
            : ((__ISwcntrl) < 24 ? (int)((1UL << (__ISwcntrl)) >> 8)
                                 : (int)((1UL << (__ISwcntrl)) >> 24)))),
  _ISwpunct =
    ((__ISwpunct) < 8
       ? (int)((1UL << (__ISwpunct)) << 24)
       : ((__ISwpunct) < 16
            ? (int)((1UL << (__ISwpunct)) << 8)
            : ((__ISwpunct) < 24 ? (int)((1UL << (__ISwpunct)) >> 8)
                                 : (int)((1UL << (__ISwpunct)) >> 24)))),
  _ISwalnum =
    ((__ISwalnum) < 8
       ? (int)((1UL << (__ISwalnum)) << 24)
       : ((__ISwalnum) < 16
            ? (int)((1UL << (__ISwalnum)) << 8)
            : ((__ISwalnum) < 24 ? (int)((1UL << (__ISwalnum)) >> 8)
                                 : (int)((1UL << (__ISwalnum)) >> 24))))
};

extern int iswalnum(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswalpha(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswcntrl(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswdigit(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswgraph(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswlower(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswprint(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswpunct(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswspace(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswupper(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswxdigit(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern int iswblank(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern wctype_t wctype(const char *__property)
  __attribute__((__nothrow__, __leaf__));
extern int iswctype(wint_t __wc, wctype_t __desc)
  __attribute__((__nothrow__, __leaf__));
extern wint_t towlower(wint_t __wc) __attribute__((__nothrow__, __leaf__));
extern wint_t towupper(wint_t __wc) __attribute__((__nothrow__, __leaf__));

typedef const __int32_t *wctrans_t;
extern wctrans_t wctrans(const char *__property)
  __attribute__((__nothrow__, __leaf__));
extern wint_t towctrans(wint_t __wc, wctrans_t __desc)
  __attribute__((__nothrow__, __leaf__));
extern int iswalnum_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswalpha_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswcntrl_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswdigit_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswgraph_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswlower_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswprint_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswpunct_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswspace_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswupper_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswxdigit_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswblank_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern wctype_t wctype_l(const char *__property, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern int iswctype_l(wint_t __wc, wctype_t __desc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern wint_t towlower_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern wint_t towupper_l(wint_t __wc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern wctrans_t wctrans_l(const char *__property, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));
extern wint_t towctrans_l(wint_t __wc, wctrans_t __desc, locale_t __locale)
  __attribute__((__nothrow__, __leaf__));

typedef __mbstate_t mbstate_t;

struct tm;
extern wchar_t *
wcscpy(wchar_t *__restrict __dest, const wchar_t *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern wchar_t *
wcsncpy(wchar_t *__restrict __dest, const wchar_t *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern wchar_t *
wcscat(wchar_t *__restrict __dest, const wchar_t *__restrict __src)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern wchar_t *
wcsncat(wchar_t *__restrict __dest, const wchar_t *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__nonnull__(1, 2)));
extern int wcscmp(const wchar_t *__s1, const wchar_t *__s2)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int wcsncmp(const wchar_t *__s1, const wchar_t *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__))
  __attribute__((__nonnull__(1, 2)));
extern int wcscasecmp(const wchar_t *__s1, const wchar_t *__s2)
  __attribute__((__nothrow__, __leaf__));
extern int wcsncasecmp(const wchar_t *__s1, const wchar_t *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern int
wcscasecmp_l(const wchar_t *__s1, const wchar_t *__s2, locale_t __loc)
  __attribute__((__nothrow__, __leaf__));
extern int wcsncasecmp_l(
  const wchar_t *__s1,
  const wchar_t *__s2,
  size_t __n,
  locale_t __loc) __attribute__((__nothrow__, __leaf__));
extern int wcscoll(const wchar_t *__s1, const wchar_t *__s2)
  __attribute__((__nothrow__, __leaf__));
extern size_t
wcsxfrm(wchar_t *__restrict __s1, const wchar_t *__restrict __s2, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern int wcscoll_l(const wchar_t *__s1, const wchar_t *__s2, locale_t __loc)
  __attribute__((__nothrow__, __leaf__));
extern size_t
wcsxfrm_l(wchar_t *__s1, const wchar_t *__s2, size_t __n, locale_t __loc)
  __attribute__((__nothrow__, __leaf__));
extern wchar_t *wcsdup(const wchar_t *__s)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern wchar_t *wcschr(const wchar_t *__wcs, wchar_t __wc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *wcsrchr(const wchar_t *__wcs, wchar_t __wc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern size_t wcscspn(const wchar_t *__wcs, const wchar_t *__reject)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern size_t wcsspn(const wchar_t *__wcs, const wchar_t *__accept)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *wcspbrk(const wchar_t *__wcs, const wchar_t *__accept)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *wcsstr(const wchar_t *__haystack, const wchar_t *__needle)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *wcstok(
  wchar_t *__restrict __s,
  const wchar_t *__restrict __delim,
  wchar_t **__restrict __ptr) __attribute__((__nothrow__, __leaf__));
extern size_t wcslen(const wchar_t *__s) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__));
extern size_t wcsnlen(const wchar_t *__s, size_t __maxlen)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *wmemchr(const wchar_t *__s, wchar_t __c, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern int wmemcmp(const wchar_t *__s1, const wchar_t *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__pure__));
extern wchar_t *
wmemcpy(wchar_t *__restrict __s1, const wchar_t *__restrict __s2, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern wchar_t *wmemmove(wchar_t *__s1, const wchar_t *__s2, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern wchar_t *wmemset(wchar_t *__s, wchar_t __c, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern wint_t btowc(int __c) __attribute__((__nothrow__, __leaf__));
extern int wctob(wint_t __c) __attribute__((__nothrow__, __leaf__));
extern int mbsinit(const mbstate_t *__ps) __attribute__((__nothrow__, __leaf__))
__attribute__((__pure__));
extern size_t mbrtowc(
  wchar_t *__restrict __pwc,
  const char *__restrict __s,
  size_t __n,
  mbstate_t *__restrict __p) __attribute__((__nothrow__, __leaf__));
extern size_t
wcrtomb(char *__restrict __s, wchar_t __wc, mbstate_t *__restrict __ps)
  __attribute__((__nothrow__, __leaf__));
extern size_t
__mbrlen(const char *__restrict __s, size_t __n, mbstate_t *__restrict __ps)
  __attribute__((__nothrow__, __leaf__));
extern size_t
mbrlen(const char *__restrict __s, size_t __n, mbstate_t *__restrict __ps)
  __attribute__((__nothrow__, __leaf__));
extern size_t mbsrtowcs(
  wchar_t *__restrict __dst,
  const char **__restrict __src,
  size_t __len,
  mbstate_t *__restrict __ps) __attribute__((__nothrow__, __leaf__));
extern size_t wcsrtombs(
  char *__restrict __dst,
  const wchar_t **__restrict __src,
  size_t __len,
  mbstate_t *__restrict __ps) __attribute__((__nothrow__, __leaf__));
extern size_t mbsnrtowcs(
  wchar_t *__restrict __dst,
  const char **__restrict __src,
  size_t __nmc,
  size_t __len,
  mbstate_t *__restrict __ps) __attribute__((__nothrow__, __leaf__));
extern size_t wcsnrtombs(
  char *__restrict __dst,
  const wchar_t **__restrict __src,
  size_t __nwc,
  size_t __len,
  mbstate_t *__restrict __ps) __attribute__((__nothrow__, __leaf__));
extern double
wcstod(const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__));
extern float
wcstof(const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__));
extern long double
wcstold(const wchar_t *__restrict __nptr, wchar_t **__restrict __endptr)
  __attribute__((__nothrow__, __leaf__));
extern long int wcstol(
  const wchar_t *__restrict __nptr,
  wchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));
extern unsigned long int wcstoul(
  const wchar_t *__restrict __nptr,
  wchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));
__extension__ extern long long int wcstoll(
  const wchar_t *__restrict __nptr,
  wchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));
__extension__ extern unsigned long long int wcstoull(
  const wchar_t *__restrict __nptr,
  wchar_t **__restrict __endptr,
  int __base) __attribute__((__nothrow__, __leaf__));
extern wchar_t *
wcpcpy(wchar_t *__restrict __dest, const wchar_t *__restrict __src)
  __attribute__((__nothrow__, __leaf__));
extern wchar_t *
wcpncpy(wchar_t *__restrict __dest, const wchar_t *__restrict __src, size_t __n)
  __attribute__((__nothrow__, __leaf__));
extern __FILE *open_wmemstream(wchar_t **__bufloc, size_t *__sizeloc)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__malloc__));
extern int fwide(__FILE *__fp, int __mode)
  __attribute__((__nothrow__, __leaf__));
extern int
fwprintf(__FILE *__restrict __stream, const wchar_t *__restrict __format, ...);
extern int wprintf(const wchar_t *__restrict __format, ...);
extern int swprintf(
  wchar_t *__restrict __s,
  size_t __n,
  const wchar_t *__restrict __format,
  ...) __attribute__((__nothrow__, __leaf__));
extern int vfwprintf(
  __FILE *__restrict __s,
  const wchar_t *__restrict __format,
  __gnuc_va_list __arg);
extern int vwprintf(const wchar_t *__restrict __format, __gnuc_va_list __arg);
extern int vswprintf(
  wchar_t *__restrict __s,
  size_t __n,
  const wchar_t *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__nothrow__, __leaf__));
extern int
fwscanf(__FILE *__restrict __stream, const wchar_t *__restrict __format, ...);
extern int wscanf(const wchar_t *__restrict __format, ...);
extern int
swscanf(const wchar_t *__restrict __s, const wchar_t *__restrict __format, ...)
  __attribute__((__nothrow__, __leaf__));
extern int
fwscanf(__FILE *__restrict __stream, const wchar_t *__restrict __format, ...) __asm__(
  ""
  "__isoc99_fwscanf");
extern int wscanf(const wchar_t *__restrict __format, ...) __asm__(
  ""
  "__isoc99_wscanf");
extern int
swscanf(const wchar_t *__restrict __s, const wchar_t *__restrict __format, ...) __asm__(
  ""
  "__isoc99_swscanf") __attribute__((__nothrow__, __leaf__));
extern int vfwscanf(
  __FILE *__restrict __s,
  const wchar_t *__restrict __format,
  __gnuc_va_list __arg);
extern int vwscanf(const wchar_t *__restrict __format, __gnuc_va_list __arg);
extern int vswscanf(
  const wchar_t *__restrict __s,
  const wchar_t *__restrict __format,
  __gnuc_va_list __arg) __attribute__((__nothrow__, __leaf__));
extern int
vfwscanf(__FILE *__restrict __s, const wchar_t *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vfwscanf");
extern int
vwscanf(const wchar_t *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vwscanf");
extern int
vswscanf(const wchar_t *__restrict __s, const wchar_t *__restrict __format, __gnuc_va_list __arg) __asm__(
  ""
  "__isoc99_vswscanf") __attribute__((__nothrow__, __leaf__));
extern wint_t fgetwc(__FILE *__stream);
extern wint_t getwc(__FILE *__stream);
extern wint_t getwchar(void);
extern wint_t fputwc(wchar_t __wc, __FILE *__stream);
extern wint_t putwc(wchar_t __wc, __FILE *__stream);
extern wint_t putwchar(wchar_t __wc);
extern wchar_t *
fgetws(wchar_t *__restrict __ws, int __n, __FILE *__restrict __stream);
extern int fputws(const wchar_t *__restrict __ws, __FILE *__restrict __stream);
extern wint_t ungetwc(wint_t __wc, __FILE *__stream);
extern size_t wcsftime(
  wchar_t *__restrict __s,
  size_t __maxsize,
  const wchar_t *__restrict __format,
  const struct tm *__restrict __tp) __attribute__((__nothrow__, __leaf__));

void printLine(const char *line)
{
  if(line != ((void *)0))
  {
    printf("%s\n", line);
  }
}
void printWLine(const wchar_t *line)
{
  if(line != ((void *)0))
  {
    wprintf(L"%ls\n", line);
  }
}
void printIntLine(int intNumber)
{
  printf("%d\n", intNumber);
}
void printShortLine(short shortNumber)
{
  printf("%hd\n", shortNumber);
}
void printFloatLine(float floatNumber)
{
  printf("%f\n", floatNumber);
}
void printLongLine(long longNumber)
{
  printf("%ld\n", longNumber);
}
void printLongLongLine(int64_t longLongIntNumber)
{
  printf(
    "%"
    "l"
    "d"
    "\n",
    longLongIntNumber);
}
void printSizeTLine(size_t sizeTNumber)
{
  printf("%zu\n", sizeTNumber);
}
void printHexCharLine(char charHex)
{
  printf("%02x\n", charHex);
}
void printWcharLine(wchar_t wideChar)
{
  wchar_t s[2];
  s[0] = wideChar;
  s[1] = L'\0';
  printf("%ls\n", s);
}
void printUnsignedLine(unsigned unsignedNumber)
{
  printf("%u\n", unsignedNumber);
}
void printHexUnsignedCharLine(unsigned char unsignedCharacter)
{
  printf("%02x\n", unsignedCharacter);
}
void printDoubleLine(double doubleNumber)
{
  printf("%g\n", doubleNumber);
}
void printStructLine(const twoIntsStruct *structTwoIntsStruct)
{
  printf(
    "%d -- %d\n", structTwoIntsStruct->intOne, structTwoIntsStruct->intTwo);
}
void printBytesLine(const unsigned char *bytes, size_t numBytes)
{
  size_t i;
  for(i = 0; i < numBytes; ++i)
  {
    printf("%02x", bytes[i]);
  }
  puts("");
}
size_t decodeHexChars(unsigned char *bytes, size_t numBytes, const char *hex)
{
  size_t numWritten = 0;
  while(numWritten < numBytes &&
        ((*__ctype_b_loc())[(int)((hex[2 * numWritten]))] &
         (unsigned short int)_ISxdigit) &&
        ((*__ctype_b_loc())[(int)((hex[2 * numWritten + 1]))] &
         (unsigned short int)_ISxdigit))
  {
    int byte;
    sscanf(&hex[2 * numWritten], "%02x", &byte);
    bytes[numWritten] = (unsigned char)byte;
    ++numWritten;
  }
  return numWritten;
}
size_t
decodeHexWChars(unsigned char *bytes, size_t numBytes, const wchar_t *hex)
{
  size_t numWritten = 0;
  while(numWritten < numBytes && iswxdigit(hex[2 * numWritten]) &&
        iswxdigit(hex[2 * numWritten + 1]))
  {
    int byte;
    swscanf(&hex[2 * numWritten], L"%02x", &byte);
    bytes[numWritten] = (unsigned char)byte;
    ++numWritten;
  }
  return numWritten;
}
int globalReturnsTrue()
{
  return 1;
}
int globalReturnsFalse()
{
  return 0;
}
int globalReturnsTrueOrFalse()
{
  return (rand() % 2);
}
const int GLOBAL_CONST_TRUE = 1;
const int GLOBAL_CONST_FALSE = 0;
const int GLOBAL_CONST_FIVE = 5;
int globalTrue = 1;
int globalFalse = 0;
int globalFive = 5;
void good1()
{
}
void good2()
{
}
void good3()
{
}
void good4()
{
}
void good5()
{
}
void good6()
{
}
void good7()
{
}
void good8()
{
}
void good9()
{
}
void bad1()
{
}
void bad2()
{
}
void bad3()
{
}
void bad4()
{
}
void bad5()
{
}
void bad6()
{
}
void bad7()
{
}
void bad8()
{
}
void bad9()
{
}
int globalArgc = 0;
char **globalArgv = ((void *)0);
void CWE190_Integer_Overflow__int_fgets_multiply_53b_goodG2BSink(int data);
static void goodG2B()
{
  int data;
  data = 0;
  data = 2;
  CWE190_Integer_Overflow__int_fgets_multiply_53b_goodG2BSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53b_goodB2GSink(int data);
static void goodB2G()
{
  int data;
  data = 0;
  {
    char inputBuffer[(3 * sizeof(data) + 2)] = "";
    if(fgets(inputBuffer, (3 * sizeof(data) + 2), stdin) != ((void *)0))
    {
      data = atoi(inputBuffer);
    }
    else
    {
      printLine("fgets() failed.");
    }
  }
  CWE190_Integer_Overflow__int_fgets_multiply_53b_goodB2GSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53_good()
{
  goodG2B();
  goodB2G();
}
int main(int argc, char *argv[])
{
  srand((unsigned)time(((void *)0)));
  printLine("Calling good()...");
  CWE190_Integer_Overflow__int_fgets_multiply_53_good();
  printLine("Finished good()");
  return 0;
}
void CWE190_Integer_Overflow__int_fgets_multiply_53c_goodG2BSink(int data);
void CWE190_Integer_Overflow__int_fgets_multiply_53b_goodG2BSink(int data)
{
  CWE190_Integer_Overflow__int_fgets_multiply_53c_goodG2BSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53c_goodB2GSink(int data);
void CWE190_Integer_Overflow__int_fgets_multiply_53b_goodB2GSink(int data)
{
  CWE190_Integer_Overflow__int_fgets_multiply_53c_goodB2GSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53d_goodG2BSink(int data);
void CWE190_Integer_Overflow__int_fgets_multiply_53c_goodG2BSink(int data)
{
  CWE190_Integer_Overflow__int_fgets_multiply_53d_goodG2BSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53d_goodB2GSink(int data);
void CWE190_Integer_Overflow__int_fgets_multiply_53c_goodB2GSink(int data)
{
  CWE190_Integer_Overflow__int_fgets_multiply_53d_goodB2GSink(data);
}
void CWE190_Integer_Overflow__int_fgets_multiply_53d_goodG2BSink(int data)
{
  if(data > 0)
  {
    int result = data * 2;
    printIntLine(result);
  }
}
void CWE190_Integer_Overflow__int_fgets_multiply_53d_goodB2GSink(int data)
{
  if(data > 0)
  {
    if(data < (0x7fffffff / 2))
    {
      int result = data * 2;
      printIntLine(result);
    }
    else
    {
      printLine("data value is too large to perform arithmetic safely.");
    }
  }
}
