// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2024 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: MIT
//
// This file is extracted from libvsync, a verified library of synchronization
// primitives and concurrent data structures. For the full library, access:
//
//     https://github.com/open-s4c/libvsync
//
// This file was automatically generated from
//
//     test/queue/bounded_mpmc_check_full.c
//
// by expanding the preprocessor macros with the following compiler flags:
//
//     -E -P -m32 --std=c99 -DVSYNC_VERIFICATION -DVSYNC_VERIFICATION_GENERIC -DVATOMIC_ENABLE_ATOMIC_SC
//
// This file may contain manually injected bugs. Please see accompanying
// `bugs.patch` file for details.
//
// Version:
//
// - libvsync 4.0.1
// - compiler GNU 9.4.0
//

typedef unsigned int size_t;
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
__extension__ typedef signed long long int __int64_t;
__extension__ typedef unsigned long long int __uint64_t;
typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;
__extension__ typedef long long int __quad_t;
__extension__ typedef unsigned long long int __u_quad_t;
__extension__ typedef long long int __intmax_t;
__extension__ typedef unsigned long long int __uintmax_t;
__extension__ typedef __uint64_t __dev_t;
__extension__ typedef unsigned int __uid_t;
__extension__ typedef unsigned int __gid_t;
__extension__ typedef unsigned long int __ino_t;
__extension__ typedef __uint64_t __ino64_t;
__extension__ typedef unsigned int __mode_t;
__extension__ typedef unsigned int __nlink_t;
__extension__ typedef long int __off_t;
__extension__ typedef __int64_t __off64_t;
__extension__ typedef int __pid_t;
__extension__ typedef struct { int __val[2]; } __fsid_t;
__extension__ typedef long int __clock_t;
__extension__ typedef unsigned long int __rlim_t;
__extension__ typedef __uint64_t __rlim64_t;
__extension__ typedef unsigned int __id_t;
__extension__ typedef long int __time_t;
__extension__ typedef unsigned int __useconds_t;
__extension__ typedef long int __suseconds_t;
__extension__ typedef int __daddr_t;
__extension__ typedef int __key_t;
__extension__ typedef int __clockid_t;
__extension__ typedef void * __timer_t;
__extension__ typedef long int __blksize_t;
__extension__ typedef long int __blkcnt_t;
__extension__ typedef __int64_t __blkcnt64_t;
__extension__ typedef unsigned long int __fsblkcnt_t;
__extension__ typedef __uint64_t __fsblkcnt64_t;
__extension__ typedef unsigned long int __fsfilcnt_t;
__extension__ typedef __uint64_t __fsfilcnt64_t;
__extension__ typedef int __fsword_t;
__extension__ typedef int __ssize_t;
__extension__ typedef long int __syscall_slong_t;
__extension__ typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef char *__caddr_t;
__extension__ typedef int __intptr_t;
__extension__ typedef unsigned int __socklen_t;
typedef int __sig_atomic_t;
__extension__ typedef __int64_t __time64_t;
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
  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
};
typedef __fpos_t fpos_t;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
extern int remove (const char *__filename) __attribute__ ((__nothrow__ , __leaf__));
extern int rename (const char *__old, const char *__new) __attribute__ ((__nothrow__ , __leaf__));
extern FILE *tmpfile (void) ;
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__ , __leaf__)) ;
extern int fclose (FILE *__stream);
extern int fflush (FILE *__stream);
extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) ;
extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) ;
extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__ , __leaf__));
extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__ , __leaf__));
extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);
extern int printf (const char *__restrict __format, ...);
extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) __attribute__ ((__nothrow__));
extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg);
extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);
extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));
extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));
extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));
extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) ;
extern int scanf (const char *__restrict __format, ...) ;
extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) __attribute__ ((__nothrow__ , __leaf__));
extern int fscanf (FILE *__restrict __stream, const char *__restrict __format, ...) __asm__ ("" "__isoc99_fscanf") ;
extern int scanf (const char *__restrict __format, ...) __asm__ ("" "__isoc99_scanf") ;
extern int sscanf (const char *__restrict __s, const char *__restrict __format, ...) __asm__ ("" "__isoc99_sscanf") __attribute__ ((__nothrow__ , __leaf__));
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__format__ (__scanf__, 2, 0)));
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vfscanf")
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vscanf")
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vsscanf") __attribute__ ((__nothrow__ , __leaf__))
     __attribute__ ((__format__ (__scanf__, 2, 0)));
extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);
extern int getchar (void);
extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);
extern int putchar (int __c);
extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;
extern char *gets (char *__s) __attribute__ ((__deprecated__));
extern int fputs (const char *__restrict __s, FILE *__restrict __stream);
extern int puts (const char *__s);
extern int ungetc (int __c, FILE *__stream);
extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);
extern int fseek (FILE *__stream, long int __off, int __whence);
extern long int ftell (FILE *__stream) ;
extern void rewind (FILE *__stream);
extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);
extern int fsetpos (FILE *__stream, const fpos_t *__pos);
extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));
extern int feof (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) ;
extern int ferror (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) ;
extern void perror (const char *__s);
extern int __uflow (FILE *);
extern int __overflow (FILE *, int);

typedef __time_t time_t;
struct timespec
{
  __time_t tv_sec;
  __syscall_slong_t tv_nsec;
};
typedef __clock_t clock_t;
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
  long int __tm_gmtoff;
  const char *__tm_zone;
};

extern clock_t clock (void) __attribute__ ((__nothrow__ , __leaf__));
extern time_t time (time_t *__timer) __attribute__ ((__nothrow__ , __leaf__));
extern double difftime (time_t __time1, time_t __time0)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
extern time_t mktime (struct tm *__tp) __attribute__ ((__nothrow__ , __leaf__));
extern size_t strftime (char *__restrict __s, size_t __maxsize,
   const char *__restrict __format,
   const struct tm *__restrict __tp) __attribute__ ((__nothrow__ , __leaf__));
extern struct tm *gmtime (const time_t *__timer) __attribute__ ((__nothrow__ , __leaf__));
extern struct tm *localtime (const time_t *__timer) __attribute__ ((__nothrow__ , __leaf__));
extern char *asctime (const struct tm *__tp) __attribute__ ((__nothrow__ , __leaf__));
extern char *ctime (const time_t *__timer) __attribute__ ((__nothrow__ , __leaf__));
extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;

typedef __pid_t pid_t;
struct sched_param
{
  int sched_priority;
};


typedef unsigned long int __cpu_mask;
typedef struct
{
  __cpu_mask __bits[1024 / (8 * sizeof (__cpu_mask))];
} cpu_set_t;

extern int __sched_cpucount (size_t __setsize, const cpu_set_t *__setp)
     __attribute__ ((__nothrow__ , __leaf__));
extern cpu_set_t *__sched_cpualloc (size_t __count) __attribute__ ((__nothrow__ , __leaf__)) ;
extern void __sched_cpufree (cpu_set_t *__set) __attribute__ ((__nothrow__ , __leaf__));


extern int sched_setparam (__pid_t __pid, const struct sched_param *__param)
     __attribute__ ((__nothrow__ , __leaf__));
extern int sched_getparam (__pid_t __pid, struct sched_param *__param) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_setscheduler (__pid_t __pid, int __policy,
          const struct sched_param *__param) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_getscheduler (__pid_t __pid) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_yield (void) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_get_priority_max (int __algorithm) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_get_priority_min (int __algorithm) __attribute__ ((__nothrow__ , __leaf__));
extern int sched_rr_get_interval (__pid_t __pid, struct timespec *__t) __attribute__ ((__nothrow__ , __leaf__));

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
  int __kind;
  unsigned int __nusers;
  __extension__ union
  {
    struct
    {
      short __espins;
      short __eelision;
    } __elision_data;
    __pthread_slist_t __list;
  };
};
struct __pthread_rwlock_arch_t
{
  unsigned int __readers;
  unsigned int __writers;
  unsigned int __wrphase_futex;
  unsigned int __writers_futex;
  unsigned int __pad3;
  unsigned int __pad4;
  unsigned char __flags;
  unsigned char __shared;
  signed char __rwelision;
  unsigned char __pad2;
  int __cur_writer;
};
struct __pthread_cond_s
{
  __extension__ union
  {
    __extension__ unsigned long long int __wseq;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __wseq32;
  };
  __extension__ union
  {
    __extension__ unsigned long long int __g1_start;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __g1_start32;
  };
  unsigned int __g_refs[2] ;
  unsigned int __g_size[2];
  unsigned int __g1_orig_size;
  unsigned int __wrefs;
  unsigned int __g_signals[2];
};
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
  char __size[36];
  long int __align;
};
typedef union pthread_attr_t pthread_attr_t;
typedef union
{
  struct __pthread_mutex_s __data;
  char __size[24];
  long int __align;
} pthread_mutex_t;
typedef union
{
  struct __pthread_cond_s __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;
typedef int __jmp_buf[6];
enum
{
  PTHREAD_CREATE_JOINABLE,
  PTHREAD_CREATE_DETACHED
};
enum
{
  PTHREAD_MUTEX_TIMED_NP,
  PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_ADAPTIVE_NP
};
enum
{
  PTHREAD_INHERIT_SCHED,
  PTHREAD_EXPLICIT_SCHED
};
enum
{
  PTHREAD_SCOPE_SYSTEM,
  PTHREAD_SCOPE_PROCESS
};
enum
{
  PTHREAD_PROCESS_PRIVATE,
  PTHREAD_PROCESS_SHARED
};
struct _pthread_cleanup_buffer
{
  void (*__routine) (void *);
  void *__arg;
  int __canceltype;
  struct _pthread_cleanup_buffer *__prev;
};
enum
{
  PTHREAD_CANCEL_ENABLE,
  PTHREAD_CANCEL_DISABLE
};
enum
{
  PTHREAD_CANCEL_DEFERRED,
  PTHREAD_CANCEL_ASYNCHRONOUS
};

extern int pthread_create (pthread_t *__restrict __newthread,
      const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));
extern void pthread_exit (void *__retval) __attribute__ ((__noreturn__));
extern int pthread_join (pthread_t __th, void **__thread_return);
extern int pthread_detach (pthread_t __th) __attribute__ ((__nothrow__ , __leaf__));
extern pthread_t pthread_self (void) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
extern int pthread_equal (pthread_t __thread1, pthread_t __thread2)
  __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
extern int pthread_attr_init (pthread_attr_t *__attr) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_destroy (pthread_attr_t *__attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getdetachstate (const pthread_attr_t *__attr,
     int *__detachstate)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setdetachstate (pthread_attr_t *__attr,
     int __detachstate)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getguardsize (const pthread_attr_t *__attr,
          size_t *__guardsize)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setguardsize (pthread_attr_t *__attr,
          size_t __guardsize)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getschedparam (const pthread_attr_t *__restrict __attr,
           struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setschedparam (pthread_attr_t *__restrict __attr,
           const struct sched_param *__restrict
           __param) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_getschedpolicy (const pthread_attr_t *__restrict
     __attr, int *__restrict __policy)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setschedpolicy (pthread_attr_t *__attr, int __policy)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getinheritsched (const pthread_attr_t *__restrict
      __attr, int *__restrict __inherit)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setinheritsched (pthread_attr_t *__attr,
      int __inherit)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getscope (const pthread_attr_t *__restrict __attr,
      int *__restrict __scope)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setscope (pthread_attr_t *__attr, int __scope)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_attr_getstackaddr (const pthread_attr_t *__restrict
          __attr, void **__restrict __stackaddr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2))) __attribute__ ((__deprecated__));
extern int pthread_attr_setstackaddr (pthread_attr_t *__attr,
          void *__stackaddr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__));
extern int pthread_attr_getstacksize (const pthread_attr_t *__restrict
          __attr, size_t *__restrict __stacksize)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_attr_setstacksize (pthread_attr_t *__attr,
          size_t __stacksize)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_setschedparam (pthread_t __target_thread, int __policy,
      const struct sched_param *__param)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3)));
extern int pthread_getschedparam (pthread_t __target_thread,
      int *__restrict __policy,
      struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2, 3)));
extern int pthread_setschedprio (pthread_t __target_thread, int __prio)
     __attribute__ ((__nothrow__ , __leaf__));
extern int pthread_once (pthread_once_t *__once_control,
    void (*__init_routine) (void)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_setcancelstate (int __state, int *__oldstate);
extern int pthread_setcanceltype (int __type, int *__oldtype);
extern int pthread_cancel (pthread_t __th);
extern void pthread_testcancel (void);
typedef struct
{
  struct
  {
    __jmp_buf __cancel_jmp_buf;
    int __mask_was_saved;
  } __cancel_jmp_buf[1];
  void *__pad[4];
} __pthread_unwind_buf_t __attribute__ ((__aligned__));
struct __pthread_cleanup_frame
{
  void (*__cancel_routine) (void *);
  void *__cancel_arg;
  int __do_it;
  int __cancel_type;
};
extern void __pthread_register_cancel (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1)));
extern void __pthread_unregister_cancel (__pthread_unwind_buf_t *__buf)
  __attribute__ ((__regparm__ (1)));
extern void __pthread_unwind_next (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1))) __attribute__ ((__noreturn__))
     __attribute__ ((__weak__))
     ;
struct __jmp_buf_tag;
extern int __sigsetjmp (struct __jmp_buf_tag *__env, int __savemask) __attribute__ ((__nothrow__));
extern int pthread_mutex_init (pthread_mutex_t *__mutex,
          const pthread_mutexattr_t *__mutexattr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutex_destroy (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutex_trylock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutex_lock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutex_unlock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutex_getprioceiling (const pthread_mutex_t *
      __restrict __mutex,
      int *__restrict __prioceiling)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_mutex_setprioceiling (pthread_mutex_t *__restrict __mutex,
      int __prioceiling,
      int *__restrict __old_ceiling)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 3)));
extern int pthread_mutexattr_init (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutexattr_destroy (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutexattr_getpshared (const pthread_mutexattr_t *
      __restrict __attr,
      int *__restrict __pshared)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_mutexattr_setpshared (pthread_mutexattr_t *__attr,
      int __pshared)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutexattr_getprotocol (const pthread_mutexattr_t *
       __restrict __attr,
       int *__restrict __protocol)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_mutexattr_setprotocol (pthread_mutexattr_t *__attr,
       int __protocol)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_mutexattr_getprioceiling (const pthread_mutexattr_t *
          __restrict __attr,
          int *__restrict __prioceiling)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_mutexattr_setprioceiling (pthread_mutexattr_t *__attr,
          int __prioceiling)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_cond_init (pthread_cond_t *__restrict __cond,
         const pthread_condattr_t *__restrict __cond_attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_cond_destroy (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_cond_signal (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_cond_broadcast (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_cond_wait (pthread_cond_t *__restrict __cond,
         pthread_mutex_t *__restrict __mutex)
     __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_cond_timedwait (pthread_cond_t *__restrict __cond,
       pthread_mutex_t *__restrict __mutex,
       const struct timespec *__restrict __abstime)
     __attribute__ ((__nonnull__ (1, 2, 3)));
extern int pthread_condattr_init (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_condattr_destroy (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_condattr_getpshared (const pthread_condattr_t *
     __restrict __attr,
     int *__restrict __pshared)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int pthread_condattr_setpshared (pthread_condattr_t *__attr,
     int __pshared) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_key_create (pthread_key_t *__key,
          void (*__destr_function) (void *))
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int pthread_key_delete (pthread_key_t __key) __attribute__ ((__nothrow__ , __leaf__));
extern void *pthread_getspecific (pthread_key_t __key) __attribute__ ((__nothrow__ , __leaf__));
extern int pthread_setspecific (pthread_key_t __key,
    const void *__pointer) __attribute__ ((__nothrow__ , __leaf__)) ;
extern int pthread_atfork (void (*__prepare) (void),
      void (*__parent) (void),
      void (*__child) (void)) __attribute__ ((__nothrow__ , __leaf__));

typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
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
typedef int int_fast16_t;
typedef int int_fast32_t;
__extension__
typedef long long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned int uint_fast16_t;
typedef unsigned int uint_fast32_t;
__extension__
typedef unsigned long long int uint_fast64_t;
typedef int intptr_t;
typedef unsigned int uintptr_t;
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;
typedef long int __gwchar_t;

typedef struct
  {
    __extension__ long long int quot;
    __extension__ long long int rem;
  } imaxdiv_t;
extern intmax_t imaxabs (intmax_t __n) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
extern imaxdiv_t imaxdiv (intmax_t __numer, intmax_t __denom)
      __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
extern intmax_t strtoimax (const char *__restrict __nptr,
      char **__restrict __endptr, int __base) __attribute__ ((__nothrow__ , __leaf__));
extern uintmax_t strtoumax (const char *__restrict __nptr,
       char ** __restrict __endptr, int __base) __attribute__ ((__nothrow__ , __leaf__));
extern intmax_t wcstoimax (const __gwchar_t *__restrict __nptr,
      __gwchar_t **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__));
extern uintmax_t wcstoumax (const __gwchar_t *__restrict __nptr,
       __gwchar_t ** __restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__));

typedef int ptrdiff_t;
typedef long int wchar_t;
typedef uint8_t vuint8_t;
typedef uint16_t vuint16_t;
typedef uint32_t vuint32_t;
typedef uint64_t vuint64_t;
typedef uintptr_t vuintptr_t;
typedef int8_t vint8_t;
typedef int16_t vint16_t;
typedef int32_t vint32_t;
typedef int64_t vint64_t;
typedef intptr_t vintptr_t;
typedef size_t vsize_t;
typedef _Bool vbool_t;

extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));

static inline void
reach_error(void)
{
    ((0) ? (void) (0) : __assert_fail ("0", "/home/diogo/Workspaces/libvsync/open/build/svcomp/include/vsync/common/assert.h", 24, __extension__ __PRETTY_FUNCTION__));
}
typedef struct vatomic8_s {
    vuint8_t _v;
} vatomic8_t;
typedef struct vatomic16_s {
    vuint16_t _v;
} __attribute__((aligned(2))) vatomic16_t;
typedef struct vatomic32_s {
    vuint32_t _v;
} __attribute__((aligned(4))) vatomic32_t;
typedef struct vatomic64_s {
    vuint64_t _v;
} __attribute__((aligned(8))) vatomic64_t;
typedef struct vatomicptr_s {
    void *_v;
} __attribute__((aligned(sizeof(void *)))) vatomicptr_t;
typedef struct vatomicsz_s {
    vsize_t _v;
} __attribute__((aligned(sizeof(vsize_t)))) vatomicsz_t;
static inline vuint32_t vatomic32_await_lt(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_lt_acq(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_lt_rlx(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_le(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_le_acq(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_le_rlx(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_gt(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_gt_acq(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_gt_rlx(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_ge(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_ge_acq(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_ge_rlx(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_neq(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_neq_acq(const vatomic32_t *a,
                                                vuint32_t v);
static inline vuint32_t vatomic32_await_neq_rlx(const vatomic32_t *a,
                                                vuint32_t v);
static inline vuint32_t vatomic32_await_eq(const vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_await_eq_acq(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_eq_rlx(const vatomic32_t *a,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_eq_add(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_eq_add_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_add_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_add_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_sub(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_eq_sub_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_sub_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_set(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_eq_set_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_set_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_eq_set_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_neq_add(vatomic32_t *a, vuint32_t c,
                                                vuint32_t v);
static inline vuint32_t vatomic32_await_neq_add_acq(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_add_rel(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_add_rlx(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_sub(vatomic32_t *a, vuint32_t c,
                                                vuint32_t v);
static inline vuint32_t vatomic32_await_neq_sub_acq(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_sub_rel(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_set(vatomic32_t *a, vuint32_t c,
                                                vuint32_t v);
static inline vuint32_t vatomic32_await_neq_set_acq(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_set_rel(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_neq_set_rlx(vatomic32_t *a, vuint32_t c,
                                                    vuint32_t v);
static inline vuint32_t vatomic32_await_lt_add(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_lt_add_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_add_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_add_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_sub(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_lt_sub_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_sub_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_set(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_lt_set_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_set_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_lt_set_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_add(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_le_add_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_add_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_add_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_sub(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_le_sub_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_sub_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_set(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_le_set_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_set_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_le_set_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_add(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_gt_add_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_add_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_add_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_sub(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_gt_sub_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_sub_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_set(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_gt_set_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_set_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_gt_set_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_add(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_ge_add_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_add_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_add_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_sub(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_ge_sub_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_sub_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_sub_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_set(vatomic32_t *a, vuint32_t c,
                                               vuint32_t v);
static inline vuint32_t vatomic32_await_ge_set_acq(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_set_rel(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint32_t vatomic32_await_ge_set_rlx(vatomic32_t *a, vuint32_t c,
                                                   vuint32_t v);
static inline vuint64_t vatomic64_await_lt(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_lt_acq(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_lt_rlx(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_le(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_le_acq(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_le_rlx(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_gt(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_gt_acq(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_gt_rlx(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_ge(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_ge_acq(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_ge_rlx(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_neq(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_neq_acq(const vatomic64_t *a,
                                                vuint64_t v);
static inline vuint64_t vatomic64_await_neq_rlx(const vatomic64_t *a,
                                                vuint64_t v);
static inline vuint64_t vatomic64_await_eq(const vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_await_eq_acq(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_eq_rlx(const vatomic64_t *a,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_eq_add(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_eq_add_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_add_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_add_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_sub(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_eq_sub_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_sub_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_set(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_eq_set_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_set_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_eq_set_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_neq_add(vatomic64_t *a, vuint64_t c,
                                                vuint64_t v);
static inline vuint64_t vatomic64_await_neq_add_acq(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_add_rel(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_add_rlx(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_sub(vatomic64_t *a, vuint64_t c,
                                                vuint64_t v);
static inline vuint64_t vatomic64_await_neq_sub_acq(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_sub_rel(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_set(vatomic64_t *a, vuint64_t c,
                                                vuint64_t v);
static inline vuint64_t vatomic64_await_neq_set_acq(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_set_rel(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_neq_set_rlx(vatomic64_t *a, vuint64_t c,
                                                    vuint64_t v);
static inline vuint64_t vatomic64_await_lt_add(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_lt_add_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_add_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_add_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_sub(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_lt_sub_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_sub_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_set(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_lt_set_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_set_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_lt_set_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_add(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_le_add_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_add_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_add_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_sub(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_le_sub_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_sub_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_set(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_le_set_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_set_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_le_set_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_add(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_gt_add_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_add_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_add_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_sub(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_gt_sub_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_sub_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_set(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_gt_set_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_set_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_gt_set_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_add(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_ge_add_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_add_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_add_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_sub(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_ge_sub_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_sub_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_sub_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_set(vatomic64_t *a, vuint64_t c,
                                               vuint64_t v);
static inline vuint64_t vatomic64_await_ge_set_acq(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_set_rel(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline vuint64_t vatomic64_await_ge_set_rlx(vatomic64_t *a, vuint64_t c,
                                                   vuint64_t v);
static inline void *vatomicptr_await_neq(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_neq_acq(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_neq_rlx(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_eq(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_eq_acq(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_eq_rlx(const vatomicptr_t *a, void *v);
static inline void *vatomicptr_await_eq_set(vatomicptr_t *a, void *c, void *v);
static inline void *vatomicptr_await_eq_set_acq(vatomicptr_t *a, void *c,
                                                void *v);
static inline void *vatomicptr_await_eq_set_rel(vatomicptr_t *a, void *c,
                                                void *v);
static inline void *vatomicptr_await_eq_set_rlx(vatomicptr_t *a, void *c,
                                                void *v);
static inline void *vatomicptr_await_neq_set(vatomicptr_t *a, void *c, void *v);
static inline void *vatomicptr_await_neq_set_acq(vatomicptr_t *a, void *c,
                                                 void *v);
static inline void *vatomicptr_await_neq_set_rel(vatomicptr_t *a, void *c,
                                                 void *v);
static inline void *vatomicptr_await_neq_set_rlx(vatomicptr_t *a, void *c,
                                                 void *v);
static inline void
verification_ignore(void)
{
}
static inline void
verification_assume(vbool_t condition)
{
    do { do { (void)(condition); do { } while (0); } while (0); } while (0);
}
static inline int
verification_rand(void)
{
    return 0;
}
static inline void
verification_loop_begin(void)
{
}
static inline void
verification_spin_start(void)
{
}
static inline void
verification_spin_end(int v)
{
    do { do { (void)(v); do { } while (0); } while (0); } while (0);
}
static inline void
verification_loop_bound(vuint32_t bound)
{
    do { do { (void)(bound); do { } while (0); } while (0); } while (0);
}
static inline void vatomic_fence(void);
static inline void vatomic_fence_acq(void);
static inline void vatomic_fence_rel(void);
static inline void vatomic_fence_rlx(void);
static inline void vatomic8_init(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_read(const vatomic8_t *a);
static inline vuint8_t vatomic8_read_acq(const vatomic8_t *a);
static inline vuint8_t vatomic8_read_rlx(const vatomic8_t *a);
static inline void vatomic8_write(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_write_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_write_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xchg(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xchg_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xchg_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xchg_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_cmpxchg(vatomic8_t *a, vuint8_t e, vuint8_t v);
static inline vuint8_t vatomic8_cmpxchg_acq(vatomic8_t *a, vuint8_t e,
                                            vuint8_t v);
static inline vuint8_t vatomic8_cmpxchg_rel(vatomic8_t *a, vuint8_t e,
                                            vuint8_t v);
static inline vuint8_t vatomic8_cmpxchg_rlx(vatomic8_t *a, vuint8_t e,
                                            vuint8_t v);
static inline vuint8_t vatomic8_get_max(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_max_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_max_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_max_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_max_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_max_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_max_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_max_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_max(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_max_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_max_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_and(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_and_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_and_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_and_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_and_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_and_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_and_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_and_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_and(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_and_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_and_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_or(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_or_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_or_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_or_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_or_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_or_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_or_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_or_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_or(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_or_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_or_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_xor(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_xor_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_xor_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_xor_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xor_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xor_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xor_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_xor_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_xor(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_xor_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_xor_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_add(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_add_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_add_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_add_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_add_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_add_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_add_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_add_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_add(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_add_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_add_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_inc(vatomic8_t *a);
static inline vuint8_t vatomic8_get_inc_acq(vatomic8_t *a);
static inline vuint8_t vatomic8_get_inc_rel(vatomic8_t *a);
static inline vuint8_t vatomic8_get_inc_rlx(vatomic8_t *a);
static inline vuint8_t vatomic8_inc_get(vatomic8_t *a);
static inline vuint8_t vatomic8_inc_get_acq(vatomic8_t *a);
static inline vuint8_t vatomic8_inc_get_rel(vatomic8_t *a);
static inline vuint8_t vatomic8_inc_get_rlx(vatomic8_t *a);
static inline void vatomic8_inc(vatomic8_t *a);
static inline void vatomic8_inc_rel(vatomic8_t *a);
static inline void vatomic8_inc_rlx(vatomic8_t *a);
static inline vuint8_t vatomic8_get_sub(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_sub_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_sub_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_sub_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_sub_get(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_sub_get_acq(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_sub_get_rel(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_sub_get_rlx(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_sub(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_sub_rel(vatomic8_t *a, vuint8_t v);
static inline void vatomic8_sub_rlx(vatomic8_t *a, vuint8_t v);
static inline vuint8_t vatomic8_get_dec(vatomic8_t *a);
static inline vuint8_t vatomic8_get_dec_acq(vatomic8_t *a);
static inline vuint8_t vatomic8_get_dec_rel(vatomic8_t *a);
static inline vuint8_t vatomic8_get_dec_rlx(vatomic8_t *a);
static inline vuint8_t vatomic8_dec_get(vatomic8_t *a);
static inline vuint8_t vatomic8_dec_get_acq(vatomic8_t *a);
static inline vuint8_t vatomic8_dec_get_rel(vatomic8_t *a);
static inline vuint8_t vatomic8_dec_get_rlx(vatomic8_t *a);
static inline void vatomic8_dec(vatomic8_t *a);
static inline void vatomic8_dec_rel(vatomic8_t *a);
static inline void vatomic8_dec_rlx(vatomic8_t *a);
static inline void vatomic16_init(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_read(const vatomic16_t *a);
static inline vuint16_t vatomic16_read_acq(const vatomic16_t *a);
static inline vuint16_t vatomic16_read_rlx(const vatomic16_t *a);
static inline void vatomic16_write(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_write_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_write_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xchg(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xchg_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xchg_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xchg_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_cmpxchg(vatomic16_t *a, vuint16_t e,
                                          vuint16_t v);
static inline vuint16_t vatomic16_cmpxchg_acq(vatomic16_t *a, vuint16_t e,
                                              vuint16_t v);
static inline vuint16_t vatomic16_cmpxchg_rel(vatomic16_t *a, vuint16_t e,
                                              vuint16_t v);
static inline vuint16_t vatomic16_cmpxchg_rlx(vatomic16_t *a, vuint16_t e,
                                              vuint16_t v);
static inline vuint16_t vatomic16_get_max(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_max_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_max_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_max_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_max_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_max_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_max_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_max_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_max(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_max_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_max_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_and(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_and_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_and_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_and_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_and_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_and_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_and_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_and_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_and(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_and_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_and_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_or(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_or_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_or_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_or_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_or_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_or_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_or_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_or_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_or(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_or_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_or_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_xor(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_xor_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_xor_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_xor_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xor_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xor_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xor_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_xor_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_xor(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_xor_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_xor_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_add(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_add_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_add_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_add_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_add_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_add_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_add_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_add_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_add(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_add_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_add_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_inc(vatomic16_t *a);
static inline vuint16_t vatomic16_get_inc_acq(vatomic16_t *a);
static inline vuint16_t vatomic16_get_inc_rel(vatomic16_t *a);
static inline vuint16_t vatomic16_get_inc_rlx(vatomic16_t *a);
static inline vuint16_t vatomic16_inc_get(vatomic16_t *a);
static inline vuint16_t vatomic16_inc_get_acq(vatomic16_t *a);
static inline vuint16_t vatomic16_inc_get_rel(vatomic16_t *a);
static inline vuint16_t vatomic16_inc_get_rlx(vatomic16_t *a);
static inline void vatomic16_inc(vatomic16_t *a);
static inline void vatomic16_inc_rel(vatomic16_t *a);
static inline void vatomic16_inc_rlx(vatomic16_t *a);
static inline vuint16_t vatomic16_get_sub(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_sub_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_sub_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_sub_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_sub_get(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_sub_get_acq(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_sub_get_rel(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_sub_get_rlx(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_sub(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_sub_rel(vatomic16_t *a, vuint16_t v);
static inline void vatomic16_sub_rlx(vatomic16_t *a, vuint16_t v);
static inline vuint16_t vatomic16_get_dec(vatomic16_t *a);
static inline vuint16_t vatomic16_get_dec_acq(vatomic16_t *a);
static inline vuint16_t vatomic16_get_dec_rel(vatomic16_t *a);
static inline vuint16_t vatomic16_get_dec_rlx(vatomic16_t *a);
static inline vuint16_t vatomic16_dec_get(vatomic16_t *a);
static inline vuint16_t vatomic16_dec_get_acq(vatomic16_t *a);
static inline vuint16_t vatomic16_dec_get_rel(vatomic16_t *a);
static inline vuint16_t vatomic16_dec_get_rlx(vatomic16_t *a);
static inline void vatomic16_dec(vatomic16_t *a);
static inline void vatomic16_dec_rel(vatomic16_t *a);
static inline void vatomic16_dec_rlx(vatomic16_t *a);
static inline void vatomic32_init(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_read(const vatomic32_t *a);
static inline vuint32_t vatomic32_read_acq(const vatomic32_t *a);
static inline vuint32_t vatomic32_read_rlx(const vatomic32_t *a);
static inline void vatomic32_write(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_write_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_write_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xchg(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xchg_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xchg_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xchg_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_cmpxchg(vatomic32_t *a, vuint32_t e,
                                          vuint32_t v);
static inline vuint32_t vatomic32_cmpxchg_acq(vatomic32_t *a, vuint32_t e,
                                              vuint32_t v);
static inline vuint32_t vatomic32_cmpxchg_rel(vatomic32_t *a, vuint32_t e,
                                              vuint32_t v);
static inline vuint32_t vatomic32_cmpxchg_rlx(vatomic32_t *a, vuint32_t e,
                                              vuint32_t v);
static inline vuint32_t vatomic32_get_max(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_max_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_max_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_max_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_max_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_max_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_max_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_max_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_max(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_max_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_max_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_and(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_and_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_and_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_and_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_and_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_and_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_and_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_and_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_and(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_and_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_and_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_or(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_or_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_or_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_or_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_or_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_or_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_or_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_or_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_or(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_or_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_or_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_xor(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_xor_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_xor_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_xor_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xor_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xor_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xor_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_xor_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_xor(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_xor_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_xor_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_add(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_add_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_add_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_add_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_add_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_add_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_add_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_add_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_add(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_add_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_add_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_inc(vatomic32_t *a);
static inline vuint32_t vatomic32_get_inc_acq(vatomic32_t *a);
static inline vuint32_t vatomic32_get_inc_rel(vatomic32_t *a);
static inline vuint32_t vatomic32_get_inc_rlx(vatomic32_t *a);
static inline vuint32_t vatomic32_inc_get(vatomic32_t *a);
static inline vuint32_t vatomic32_inc_get_acq(vatomic32_t *a);
static inline vuint32_t vatomic32_inc_get_rel(vatomic32_t *a);
static inline vuint32_t vatomic32_inc_get_rlx(vatomic32_t *a);
static inline void vatomic32_inc(vatomic32_t *a);
static inline void vatomic32_inc_rel(vatomic32_t *a);
static inline void vatomic32_inc_rlx(vatomic32_t *a);
static inline vuint32_t vatomic32_get_sub(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_sub_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_sub_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_sub_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_sub_get(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_sub_get_acq(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_sub_get_rel(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_sub_get_rlx(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_sub(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_sub_rel(vatomic32_t *a, vuint32_t v);
static inline void vatomic32_sub_rlx(vatomic32_t *a, vuint32_t v);
static inline vuint32_t vatomic32_get_dec(vatomic32_t *a);
static inline vuint32_t vatomic32_get_dec_acq(vatomic32_t *a);
static inline vuint32_t vatomic32_get_dec_rel(vatomic32_t *a);
static inline vuint32_t vatomic32_get_dec_rlx(vatomic32_t *a);
static inline vuint32_t vatomic32_dec_get(vatomic32_t *a);
static inline vuint32_t vatomic32_dec_get_acq(vatomic32_t *a);
static inline vuint32_t vatomic32_dec_get_rel(vatomic32_t *a);
static inline vuint32_t vatomic32_dec_get_rlx(vatomic32_t *a);
static inline void vatomic32_dec(vatomic32_t *a);
static inline void vatomic32_dec_rel(vatomic32_t *a);
static inline void vatomic32_dec_rlx(vatomic32_t *a);
static inline void vatomic64_init(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_read(const vatomic64_t *a);
static inline vuint64_t vatomic64_read_acq(const vatomic64_t *a);
static inline vuint64_t vatomic64_read_rlx(const vatomic64_t *a);
static inline void vatomic64_write(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_write_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_write_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xchg(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xchg_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xchg_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xchg_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_cmpxchg(vatomic64_t *a, vuint64_t e,
                                          vuint64_t v);
static inline vuint64_t vatomic64_cmpxchg_acq(vatomic64_t *a, vuint64_t e,
                                              vuint64_t v);
static inline vuint64_t vatomic64_cmpxchg_rel(vatomic64_t *a, vuint64_t e,
                                              vuint64_t v);
static inline vuint64_t vatomic64_cmpxchg_rlx(vatomic64_t *a, vuint64_t e,
                                              vuint64_t v);
static inline vuint64_t vatomic64_get_max(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_max_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_max_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_max_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_max_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_max_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_max_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_max_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_max(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_max_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_max_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_and(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_and_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_and_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_and_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_and_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_and_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_and_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_and_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_and(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_and_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_and_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_or(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_or_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_or_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_or_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_or_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_or_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_or_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_or_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_or(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_or_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_or_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_xor(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_xor_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_xor_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_xor_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xor_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xor_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xor_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_xor_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_xor(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_xor_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_xor_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_add(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_add_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_add_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_add_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_add_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_add_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_add_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_add_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_add(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_add_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_add_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_inc(vatomic64_t *a);
static inline vuint64_t vatomic64_get_inc_acq(vatomic64_t *a);
static inline vuint64_t vatomic64_get_inc_rel(vatomic64_t *a);
static inline vuint64_t vatomic64_get_inc_rlx(vatomic64_t *a);
static inline vuint64_t vatomic64_inc_get(vatomic64_t *a);
static inline vuint64_t vatomic64_inc_get_acq(vatomic64_t *a);
static inline vuint64_t vatomic64_inc_get_rel(vatomic64_t *a);
static inline vuint64_t vatomic64_inc_get_rlx(vatomic64_t *a);
static inline void vatomic64_inc(vatomic64_t *a);
static inline void vatomic64_inc_rel(vatomic64_t *a);
static inline void vatomic64_inc_rlx(vatomic64_t *a);
static inline vuint64_t vatomic64_get_sub(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_sub_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_sub_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_sub_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_sub_get(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_sub_get_acq(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_sub_get_rel(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_sub_get_rlx(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_sub(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_sub_rel(vatomic64_t *a, vuint64_t v);
static inline void vatomic64_sub_rlx(vatomic64_t *a, vuint64_t v);
static inline vuint64_t vatomic64_get_dec(vatomic64_t *a);
static inline vuint64_t vatomic64_get_dec_acq(vatomic64_t *a);
static inline vuint64_t vatomic64_get_dec_rel(vatomic64_t *a);
static inline vuint64_t vatomic64_get_dec_rlx(vatomic64_t *a);
static inline vuint64_t vatomic64_dec_get(vatomic64_t *a);
static inline vuint64_t vatomic64_dec_get_acq(vatomic64_t *a);
static inline vuint64_t vatomic64_dec_get_rel(vatomic64_t *a);
static inline vuint64_t vatomic64_dec_get_rlx(vatomic64_t *a);
static inline void vatomic64_dec(vatomic64_t *a);
static inline void vatomic64_dec_rel(vatomic64_t *a);
static inline void vatomic64_dec_rlx(vatomic64_t *a);
static inline void vatomicsz_init(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_read(const vatomicsz_t *a);
static inline vsize_t vatomicsz_read_acq(const vatomicsz_t *a);
static inline vsize_t vatomicsz_read_rlx(const vatomicsz_t *a);
static inline void vatomicsz_write(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_write_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_write_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xchg(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xchg_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xchg_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xchg_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_cmpxchg(vatomicsz_t *a, vsize_t e, vsize_t v);
static inline vsize_t vatomicsz_cmpxchg_acq(vatomicsz_t *a, vsize_t e,
                                            vsize_t v);
static inline vsize_t vatomicsz_cmpxchg_rel(vatomicsz_t *a, vsize_t e,
                                            vsize_t v);
static inline vsize_t vatomicsz_cmpxchg_rlx(vatomicsz_t *a, vsize_t e,
                                            vsize_t v);
static inline vsize_t vatomicsz_get_max(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_max_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_max_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_max_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_max_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_max_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_max_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_max_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_max(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_max_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_max_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_and(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_and_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_and_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_and_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_and_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_and_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_and_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_and_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_and(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_and_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_and_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_or(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_or_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_or_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_or_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_or_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_or_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_or_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_or_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_or(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_or_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_or_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_xor(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_xor_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_xor_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_xor_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xor_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xor_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xor_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_xor_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_xor(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_xor_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_xor_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_add(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_add_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_add_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_add_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_add_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_add_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_add_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_add_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_add(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_add_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_add_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_inc(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_inc_acq(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_inc_rel(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_inc_rlx(vatomicsz_t *a);
static inline vsize_t vatomicsz_inc_get(vatomicsz_t *a);
static inline vsize_t vatomicsz_inc_get_acq(vatomicsz_t *a);
static inline vsize_t vatomicsz_inc_get_rel(vatomicsz_t *a);
static inline vsize_t vatomicsz_inc_get_rlx(vatomicsz_t *a);
static inline void vatomicsz_inc(vatomicsz_t *a);
static inline void vatomicsz_inc_rel(vatomicsz_t *a);
static inline void vatomicsz_inc_rlx(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_sub(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_sub_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_sub_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_sub_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_sub_get(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_sub_get_acq(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_sub_get_rel(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_sub_get_rlx(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_sub(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_sub_rel(vatomicsz_t *a, vsize_t v);
static inline void vatomicsz_sub_rlx(vatomicsz_t *a, vsize_t v);
static inline vsize_t vatomicsz_get_dec(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_dec_acq(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_dec_rel(vatomicsz_t *a);
static inline vsize_t vatomicsz_get_dec_rlx(vatomicsz_t *a);
static inline vsize_t vatomicsz_dec_get(vatomicsz_t *a);
static inline vsize_t vatomicsz_dec_get_acq(vatomicsz_t *a);
static inline vsize_t vatomicsz_dec_get_rel(vatomicsz_t *a);
static inline vsize_t vatomicsz_dec_get_rlx(vatomicsz_t *a);
static inline void vatomicsz_dec(vatomicsz_t *a);
static inline void vatomicsz_dec_rel(vatomicsz_t *a);
static inline void vatomicsz_dec_rlx(vatomicsz_t *a);
static inline void vatomicptr_init(vatomicptr_t *a, void *v);
static inline void *vatomicptr_read(const vatomicptr_t *a);
static inline void *vatomicptr_read_acq(const vatomicptr_t *a);
static inline void *vatomicptr_read_rlx(const vatomicptr_t *a);
static inline void vatomicptr_write(vatomicptr_t *a, void *v);
static inline void vatomicptr_write_rel(vatomicptr_t *a, void *v);
static inline void vatomicptr_write_rlx(vatomicptr_t *a, void *v);
static inline void *vatomicptr_xchg(vatomicptr_t *a, void *v);
static inline void *vatomicptr_xchg_acq(vatomicptr_t *a, void *v);
static inline void *vatomicptr_xchg_rel(vatomicptr_t *a, void *v);
static inline void *vatomicptr_xchg_rlx(vatomicptr_t *a, void *v);
static inline void *vatomicptr_cmpxchg(vatomicptr_t *a, void *e, void *v);
static inline void *vatomicptr_cmpxchg_acq(vatomicptr_t *a, void *e, void *v);
static inline void *vatomicptr_cmpxchg_rel(vatomicptr_t *a, void *e, void *v);
static inline void *vatomicptr_cmpxchg_rlx(vatomicptr_t *a, void *e, void *v);
static inline void
vatomic_fence_rlx(void)
{
    vatomic_fence();
}
static inline void
vatomic_fence_acq(void)
{
    vatomic_fence();
}
static inline void
vatomic_fence_rel(void)
{
    vatomic_fence();
}
static inline vuint8_t
vatomic8_read_rlx(const vatomic8_t *a)
{
    return vatomic8_read(a);
}
static inline vuint8_t
vatomic8_read_acq(const vatomic8_t *a)
{
    return vatomic8_read(a);
}
static inline void
vatomic8_write_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_write(a, v);
}
static inline void
vatomic8_write_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_write(a, v);
}
static inline vuint8_t
vatomic8_xchg_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xchg(a, v);
}
static inline vuint8_t
vatomic8_xchg_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xchg(a, v);
}
static inline vuint8_t
vatomic8_xchg_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xchg(a, v);
}
static inline vuint8_t
vatomic8_cmpxchg_rlx(vatomic8_t *a, vuint8_t e, vuint8_t v)
{
    return vatomic8_cmpxchg(a, e, v);
}
static inline vuint8_t
vatomic8_cmpxchg_acq(vatomic8_t *a, vuint8_t e, vuint8_t v)
{
    return vatomic8_cmpxchg(a, e, v);
}
static inline vuint8_t
vatomic8_cmpxchg_rel(vatomic8_t *a, vuint8_t e, vuint8_t v)
{
    return vatomic8_cmpxchg(a, e, v);
}
static inline vuint8_t
vatomic8_get_max_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_max(a, v);
}
static inline vuint8_t
vatomic8_get_and_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_and(a, v);
}
static inline vuint8_t
vatomic8_get_or_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_or(a, v);
}
static inline vuint8_t
vatomic8_get_xor_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_xor(a, v);
}
static inline vuint8_t
vatomic8_get_add_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_add(a, v);
}
static inline vuint8_t
vatomic8_get_sub_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_sub(a, v);
}
static inline vuint8_t
vatomic8_max_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_max_get(a, v);
}
static inline vuint8_t
vatomic8_and_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_and_get(a, v);
}
static inline vuint8_t
vatomic8_or_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_or_get(a, v);
}
static inline vuint8_t
vatomic8_xor_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xor_get(a, v);
}
static inline vuint8_t
vatomic8_add_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_add_get(a, v);
}
static inline vuint8_t
vatomic8_sub_get_rlx(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_sub_get(a, v);
}
static inline vuint8_t
vatomic8_get_max_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_max(a, v);
}
static inline vuint8_t
vatomic8_get_and_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_and(a, v);
}
static inline vuint8_t
vatomic8_get_or_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_or(a, v);
}
static inline vuint8_t
vatomic8_get_xor_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_xor(a, v);
}
static inline vuint8_t
vatomic8_get_add_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_add(a, v);
}
static inline vuint8_t
vatomic8_get_sub_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_sub(a, v);
}
static inline vuint8_t
vatomic8_max_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_max_get(a, v);
}
static inline vuint8_t
vatomic8_and_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_and_get(a, v);
}
static inline vuint8_t
vatomic8_or_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_or_get(a, v);
}
static inline vuint8_t
vatomic8_xor_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xor_get(a, v);
}
static inline vuint8_t
vatomic8_add_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_add_get(a, v);
}
static inline vuint8_t
vatomic8_sub_get_acq(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_sub_get(a, v);
}
static inline vuint8_t
vatomic8_get_max_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_max(a, v);
}
static inline vuint8_t
vatomic8_get_and_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_and(a, v);
}
static inline vuint8_t
vatomic8_get_or_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_or(a, v);
}
static inline vuint8_t
vatomic8_get_xor_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_xor(a, v);
}
static inline vuint8_t
vatomic8_get_add_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_add(a, v);
}
static inline vuint8_t
vatomic8_get_sub_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_sub(a, v);
}
static inline vuint8_t
vatomic8_max_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_max_get(a, v);
}
static inline vuint8_t
vatomic8_and_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_and_get(a, v);
}
static inline vuint8_t
vatomic8_or_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_or_get(a, v);
}
static inline vuint8_t
vatomic8_xor_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_xor_get(a, v);
}
static inline vuint8_t
vatomic8_add_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_add_get(a, v);
}
static inline vuint8_t
vatomic8_sub_get_rel(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_sub_get(a, v);
}
static inline vuint8_t
vatomic8_get_inc_rlx(vatomic8_t *a)
{
    return vatomic8_get_inc(a);
}
static inline vuint8_t
vatomic8_inc_get_rlx(vatomic8_t *a)
{
    return vatomic8_inc_get(a);
}
static inline vuint8_t
vatomic8_get_dec_rlx(vatomic8_t *a)
{
    return vatomic8_get_dec(a);
}
static inline vuint8_t
vatomic8_dec_get_rlx(vatomic8_t *a)
{
    return vatomic8_dec_get(a);
}
static inline vuint8_t
vatomic8_get_inc_acq(vatomic8_t *a)
{
    return vatomic8_get_inc(a);
}
static inline vuint8_t
vatomic8_inc_get_acq(vatomic8_t *a)
{
    return vatomic8_inc_get(a);
}
static inline vuint8_t
vatomic8_get_dec_acq(vatomic8_t *a)
{
    return vatomic8_get_dec(a);
}
static inline vuint8_t
vatomic8_dec_get_acq(vatomic8_t *a)
{
    return vatomic8_dec_get(a);
}
static inline vuint8_t
vatomic8_get_inc_rel(vatomic8_t *a)
{
    return vatomic8_get_inc(a);
}
static inline vuint8_t
vatomic8_inc_get_rel(vatomic8_t *a)
{
    return vatomic8_inc_get(a);
}
static inline vuint8_t
vatomic8_get_dec_rel(vatomic8_t *a)
{
    return vatomic8_get_dec(a);
}
static inline vuint8_t
vatomic8_dec_get_rel(vatomic8_t *a)
{
    return vatomic8_dec_get(a);
}
static inline void
vatomic8_max_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_max(a, v);
}
static inline void
vatomic8_and_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_and(a, v);
}
static inline void
vatomic8_or_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_or(a, v);
}
static inline void
vatomic8_xor_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_xor(a, v);
}
static inline void
vatomic8_add_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_add(a, v);
}
static inline void
vatomic8_sub_rlx(vatomic8_t *a, vuint8_t v)
{
    vatomic8_sub(a, v);
}
static inline void
vatomic8_max_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_max(a, v);
}
static inline void
vatomic8_and_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_and(a, v);
}
static inline void
vatomic8_or_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_or(a, v);
}
static inline void
vatomic8_xor_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_xor(a, v);
}
static inline void
vatomic8_add_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_add(a, v);
}
static inline void
vatomic8_sub_rel(vatomic8_t *a, vuint8_t v)
{
    vatomic8_sub(a, v);
}
static inline void
vatomic8_inc_rlx(vatomic8_t *a)
{
    vatomic8_inc(a);
}
static inline void
vatomic8_dec_rlx(vatomic8_t *a)
{
    vatomic8_dec(a);
}
static inline void
vatomic8_inc_rel(vatomic8_t *a)
{
    vatomic8_inc(a);
}
static inline void
vatomic8_dec_rel(vatomic8_t *a)
{
    vatomic8_dec(a);
}
static inline vuint16_t
vatomic16_read_rlx(const vatomic16_t *a)
{
    return vatomic16_read(a);
}
static inline vuint16_t
vatomic16_read_acq(const vatomic16_t *a)
{
    return vatomic16_read(a);
}
static inline void
vatomic16_write_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_write(a, v);
}
static inline void
vatomic16_write_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_write(a, v);
}
static inline vuint16_t
vatomic16_xchg_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xchg(a, v);
}
static inline vuint16_t
vatomic16_xchg_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xchg(a, v);
}
static inline vuint16_t
vatomic16_xchg_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xchg(a, v);
}
static inline vuint16_t
vatomic16_cmpxchg_rlx(vatomic16_t *a, vuint16_t e, vuint16_t v)
{
    return vatomic16_cmpxchg(a, e, v);
}
static inline vuint16_t
vatomic16_cmpxchg_acq(vatomic16_t *a, vuint16_t e, vuint16_t v)
{
    return vatomic16_cmpxchg(a, e, v);
}
static inline vuint16_t
vatomic16_cmpxchg_rel(vatomic16_t *a, vuint16_t e, vuint16_t v)
{
    return vatomic16_cmpxchg(a, e, v);
}
static inline vuint16_t
vatomic16_get_max_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_max(a, v);
}
static inline vuint16_t
vatomic16_get_and_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_and(a, v);
}
static inline vuint16_t
vatomic16_get_or_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_or(a, v);
}
static inline vuint16_t
vatomic16_get_xor_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_xor(a, v);
}
static inline vuint16_t
vatomic16_get_add_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_add(a, v);
}
static inline vuint16_t
vatomic16_get_sub_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_sub(a, v);
}
static inline vuint16_t
vatomic16_max_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_max_get(a, v);
}
static inline vuint16_t
vatomic16_and_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_and_get(a, v);
}
static inline vuint16_t
vatomic16_or_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_or_get(a, v);
}
static inline vuint16_t
vatomic16_xor_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xor_get(a, v);
}
static inline vuint16_t
vatomic16_add_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_add_get(a, v);
}
static inline vuint16_t
vatomic16_sub_get_rlx(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_sub_get(a, v);
}
static inline vuint16_t
vatomic16_get_max_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_max(a, v);
}
static inline vuint16_t
vatomic16_get_and_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_and(a, v);
}
static inline vuint16_t
vatomic16_get_or_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_or(a, v);
}
static inline vuint16_t
vatomic16_get_xor_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_xor(a, v);
}
static inline vuint16_t
vatomic16_get_add_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_add(a, v);
}
static inline vuint16_t
vatomic16_get_sub_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_sub(a, v);
}
static inline vuint16_t
vatomic16_max_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_max_get(a, v);
}
static inline vuint16_t
vatomic16_and_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_and_get(a, v);
}
static inline vuint16_t
vatomic16_or_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_or_get(a, v);
}
static inline vuint16_t
vatomic16_xor_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xor_get(a, v);
}
static inline vuint16_t
vatomic16_add_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_add_get(a, v);
}
static inline vuint16_t
vatomic16_sub_get_acq(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_sub_get(a, v);
}
static inline vuint16_t
vatomic16_get_max_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_max(a, v);
}
static inline vuint16_t
vatomic16_get_and_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_and(a, v);
}
static inline vuint16_t
vatomic16_get_or_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_or(a, v);
}
static inline vuint16_t
vatomic16_get_xor_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_xor(a, v);
}
static inline vuint16_t
vatomic16_get_add_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_add(a, v);
}
static inline vuint16_t
vatomic16_get_sub_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_sub(a, v);
}
static inline vuint16_t
vatomic16_max_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_max_get(a, v);
}
static inline vuint16_t
vatomic16_and_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_and_get(a, v);
}
static inline vuint16_t
vatomic16_or_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_or_get(a, v);
}
static inline vuint16_t
vatomic16_xor_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_xor_get(a, v);
}
static inline vuint16_t
vatomic16_add_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_add_get(a, v);
}
static inline vuint16_t
vatomic16_sub_get_rel(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_sub_get(a, v);
}
static inline vuint16_t
vatomic16_get_inc_rlx(vatomic16_t *a)
{
    return vatomic16_get_inc(a);
}
static inline vuint16_t
vatomic16_inc_get_rlx(vatomic16_t *a)
{
    return vatomic16_inc_get(a);
}
static inline vuint16_t
vatomic16_get_dec_rlx(vatomic16_t *a)
{
    return vatomic16_get_dec(a);
}
static inline vuint16_t
vatomic16_dec_get_rlx(vatomic16_t *a)
{
    return vatomic16_dec_get(a);
}
static inline vuint16_t
vatomic16_get_inc_acq(vatomic16_t *a)
{
    return vatomic16_get_inc(a);
}
static inline vuint16_t
vatomic16_inc_get_acq(vatomic16_t *a)
{
    return vatomic16_inc_get(a);
}
static inline vuint16_t
vatomic16_get_dec_acq(vatomic16_t *a)
{
    return vatomic16_get_dec(a);
}
static inline vuint16_t
vatomic16_dec_get_acq(vatomic16_t *a)
{
    return vatomic16_dec_get(a);
}
static inline vuint16_t
vatomic16_get_inc_rel(vatomic16_t *a)
{
    return vatomic16_get_inc(a);
}
static inline vuint16_t
vatomic16_inc_get_rel(vatomic16_t *a)
{
    return vatomic16_inc_get(a);
}
static inline vuint16_t
vatomic16_get_dec_rel(vatomic16_t *a)
{
    return vatomic16_get_dec(a);
}
static inline vuint16_t
vatomic16_dec_get_rel(vatomic16_t *a)
{
    return vatomic16_dec_get(a);
}
static inline void
vatomic16_max_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_max(a, v);
}
static inline void
vatomic16_and_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_and(a, v);
}
static inline void
vatomic16_or_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_or(a, v);
}
static inline void
vatomic16_xor_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_xor(a, v);
}
static inline void
vatomic16_add_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_add(a, v);
}
static inline void
vatomic16_sub_rlx(vatomic16_t *a, vuint16_t v)
{
    vatomic16_sub(a, v);
}
static inline void
vatomic16_max_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_max(a, v);
}
static inline void
vatomic16_and_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_and(a, v);
}
static inline void
vatomic16_or_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_or(a, v);
}
static inline void
vatomic16_xor_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_xor(a, v);
}
static inline void
vatomic16_add_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_add(a, v);
}
static inline void
vatomic16_sub_rel(vatomic16_t *a, vuint16_t v)
{
    vatomic16_sub(a, v);
}
static inline void
vatomic16_inc_rlx(vatomic16_t *a)
{
    vatomic16_inc(a);
}
static inline void
vatomic16_dec_rlx(vatomic16_t *a)
{
    vatomic16_dec(a);
}
static inline void
vatomic16_inc_rel(vatomic16_t *a)
{
    vatomic16_inc(a);
}
static inline void
vatomic16_dec_rel(vatomic16_t *a)
{
    vatomic16_dec(a);
}
static inline vuint32_t
vatomic32_read_rlx(const vatomic32_t *a)
{
    return vatomic32_read(a);
}
static inline vuint32_t
vatomic32_read_acq(const vatomic32_t *a)
{
    return vatomic32_read(a);
}
static inline void
vatomic32_write_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_write(a, v);
}
static inline void
vatomic32_write_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_write(a, v);
}
static inline vuint32_t
vatomic32_xchg_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xchg(a, v);
}
static inline vuint32_t
vatomic32_xchg_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xchg(a, v);
}
static inline vuint32_t
vatomic32_xchg_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xchg(a, v);
}
static inline vuint32_t
vatomic32_cmpxchg_rlx(vatomic32_t *a, vuint32_t e, vuint32_t v)
{
    return vatomic32_cmpxchg(a, e, v);
}
static inline vuint32_t
vatomic32_cmpxchg_acq(vatomic32_t *a, vuint32_t e, vuint32_t v)
{
    return vatomic32_cmpxchg(a, e, v);
}
static inline vuint32_t
vatomic32_cmpxchg_rel(vatomic32_t *a, vuint32_t e, vuint32_t v)
{
    return vatomic32_cmpxchg(a, e, v);
}
static inline vuint32_t
vatomic32_get_max_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_max(a, v);
}
static inline vuint32_t
vatomic32_get_and_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_and(a, v);
}
static inline vuint32_t
vatomic32_get_or_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_or(a, v);
}
static inline vuint32_t
vatomic32_get_xor_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_xor(a, v);
}
static inline vuint32_t
vatomic32_get_add_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_add(a, v);
}
static inline vuint32_t
vatomic32_get_sub_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_sub(a, v);
}
static inline vuint32_t
vatomic32_max_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_max_get(a, v);
}
static inline vuint32_t
vatomic32_and_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_and_get(a, v);
}
static inline vuint32_t
vatomic32_or_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_or_get(a, v);
}
static inline vuint32_t
vatomic32_xor_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xor_get(a, v);
}
static inline vuint32_t
vatomic32_add_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_add_get(a, v);
}
static inline vuint32_t
vatomic32_sub_get_rlx(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_sub_get(a, v);
}
static inline vuint32_t
vatomic32_get_max_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_max(a, v);
}
static inline vuint32_t
vatomic32_get_and_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_and(a, v);
}
static inline vuint32_t
vatomic32_get_or_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_or(a, v);
}
static inline vuint32_t
vatomic32_get_xor_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_xor(a, v);
}
static inline vuint32_t
vatomic32_get_add_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_add(a, v);
}
static inline vuint32_t
vatomic32_get_sub_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_sub(a, v);
}
static inline vuint32_t
vatomic32_max_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_max_get(a, v);
}
static inline vuint32_t
vatomic32_and_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_and_get(a, v);
}
static inline vuint32_t
vatomic32_or_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_or_get(a, v);
}
static inline vuint32_t
vatomic32_xor_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xor_get(a, v);
}
static inline vuint32_t
vatomic32_add_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_add_get(a, v);
}
static inline vuint32_t
vatomic32_sub_get_acq(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_sub_get(a, v);
}
static inline vuint32_t
vatomic32_get_max_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_max(a, v);
}
static inline vuint32_t
vatomic32_get_and_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_and(a, v);
}
static inline vuint32_t
vatomic32_get_or_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_or(a, v);
}
static inline vuint32_t
vatomic32_get_xor_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_xor(a, v);
}
static inline vuint32_t
vatomic32_get_add_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_add(a, v);
}
static inline vuint32_t
vatomic32_get_sub_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_sub(a, v);
}
static inline vuint32_t
vatomic32_max_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_max_get(a, v);
}
static inline vuint32_t
vatomic32_and_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_and_get(a, v);
}
static inline vuint32_t
vatomic32_or_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_or_get(a, v);
}
static inline vuint32_t
vatomic32_xor_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_xor_get(a, v);
}
static inline vuint32_t
vatomic32_add_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_add_get(a, v);
}
static inline vuint32_t
vatomic32_sub_get_rel(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_sub_get(a, v);
}
static inline vuint32_t
vatomic32_get_inc_rlx(vatomic32_t *a)
{
    return vatomic32_get_inc(a);
}
static inline vuint32_t
vatomic32_inc_get_rlx(vatomic32_t *a)
{
    return vatomic32_inc_get(a);
}
static inline vuint32_t
vatomic32_get_dec_rlx(vatomic32_t *a)
{
    return vatomic32_get_dec(a);
}
static inline vuint32_t
vatomic32_dec_get_rlx(vatomic32_t *a)
{
    return vatomic32_dec_get(a);
}
static inline vuint32_t
vatomic32_get_inc_acq(vatomic32_t *a)
{
    return vatomic32_get_inc(a);
}
static inline vuint32_t
vatomic32_inc_get_acq(vatomic32_t *a)
{
    return vatomic32_inc_get(a);
}
static inline vuint32_t
vatomic32_get_dec_acq(vatomic32_t *a)
{
    return vatomic32_get_dec(a);
}
static inline vuint32_t
vatomic32_dec_get_acq(vatomic32_t *a)
{
    return vatomic32_dec_get(a);
}
static inline vuint32_t
vatomic32_get_inc_rel(vatomic32_t *a)
{
    return vatomic32_get_inc(a);
}
static inline vuint32_t
vatomic32_inc_get_rel(vatomic32_t *a)
{
    return vatomic32_inc_get(a);
}
static inline vuint32_t
vatomic32_get_dec_rel(vatomic32_t *a)
{
    return vatomic32_get_dec(a);
}
static inline vuint32_t
vatomic32_dec_get_rel(vatomic32_t *a)
{
    return vatomic32_dec_get(a);
}
static inline void
vatomic32_max_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_max(a, v);
}
static inline void
vatomic32_and_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_and(a, v);
}
static inline void
vatomic32_or_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_or(a, v);
}
static inline void
vatomic32_xor_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_xor(a, v);
}
static inline void
vatomic32_add_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_add(a, v);
}
static inline void
vatomic32_sub_rlx(vatomic32_t *a, vuint32_t v)
{
    vatomic32_sub(a, v);
}
static inline void
vatomic32_max_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_max(a, v);
}
static inline void
vatomic32_and_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_and(a, v);
}
static inline void
vatomic32_or_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_or(a, v);
}
static inline void
vatomic32_xor_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_xor(a, v);
}
static inline void
vatomic32_add_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_add(a, v);
}
static inline void
vatomic32_sub_rel(vatomic32_t *a, vuint32_t v)
{
    vatomic32_sub(a, v);
}
static inline void
vatomic32_inc_rlx(vatomic32_t *a)
{
    vatomic32_inc(a);
}
static inline void
vatomic32_dec_rlx(vatomic32_t *a)
{
    vatomic32_dec(a);
}
static inline void
vatomic32_inc_rel(vatomic32_t *a)
{
    vatomic32_inc(a);
}
static inline void
vatomic32_dec_rel(vatomic32_t *a)
{
    vatomic32_dec(a);
}
static inline vuint32_t
vatomic32_await_eq_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_eq(a, v);
}
static inline vuint32_t
vatomic32_await_neq_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_neq(a, v);
}
static inline vuint32_t
vatomic32_await_lt_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_lt(a, v);
}
static inline vuint32_t
vatomic32_await_le_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_le(a, v);
}
static inline vuint32_t
vatomic32_await_gt_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_gt(a, v);
}
static inline vuint32_t
vatomic32_await_ge_rlx(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_ge(a, v);
}
static inline vuint32_t
vatomic32_await_eq_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_eq(a, v);
}
static inline vuint32_t
vatomic32_await_neq_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_neq(a, v);
}
static inline vuint32_t
vatomic32_await_lt_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_lt(a, v);
}
static inline vuint32_t
vatomic32_await_le_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_le(a, v);
}
static inline vuint32_t
vatomic32_await_gt_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_gt(a, v);
}
static inline vuint32_t
vatomic32_await_ge_acq(const vatomic32_t *a, vuint32_t v)
{
    return vatomic32_await_ge(a, v);
}
static inline vuint32_t
vatomic32_await_le_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_le_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_le_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_lt_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_lt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_ge_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_ge_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_gt_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_gt_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_neq_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_neq_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_add_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_add_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_add_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_add(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_sub_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_sub_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_sub_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_sub(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_set_rlx(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_set_acq(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_set(a, c, v);
}
static inline vuint32_t
vatomic32_await_eq_set_rel(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    return vatomic32_await_eq_set(a, c, v);
}
static inline vuint64_t
vatomic64_read_rlx(const vatomic64_t *a)
{
    return vatomic64_read(a);
}
static inline vuint64_t
vatomic64_read_acq(const vatomic64_t *a)
{
    return vatomic64_read(a);
}
static inline void
vatomic64_write_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_write(a, v);
}
static inline void
vatomic64_write_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_write(a, v);
}
static inline vuint64_t
vatomic64_xchg_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xchg(a, v);
}
static inline vuint64_t
vatomic64_xchg_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xchg(a, v);
}
static inline vuint64_t
vatomic64_xchg_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xchg(a, v);
}
static inline vuint64_t
vatomic64_cmpxchg_rlx(vatomic64_t *a, vuint64_t e, vuint64_t v)
{
    return vatomic64_cmpxchg(a, e, v);
}
static inline vuint64_t
vatomic64_cmpxchg_acq(vatomic64_t *a, vuint64_t e, vuint64_t v)
{
    return vatomic64_cmpxchg(a, e, v);
}
static inline vuint64_t
vatomic64_cmpxchg_rel(vatomic64_t *a, vuint64_t e, vuint64_t v)
{
    return vatomic64_cmpxchg(a, e, v);
}
static inline vuint64_t
vatomic64_get_max_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_max(a, v);
}
static inline vuint64_t
vatomic64_get_and_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_and(a, v);
}
static inline vuint64_t
vatomic64_get_or_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_or(a, v);
}
static inline vuint64_t
vatomic64_get_xor_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_xor(a, v);
}
static inline vuint64_t
vatomic64_get_add_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_add(a, v);
}
static inline vuint64_t
vatomic64_get_sub_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_sub(a, v);
}
static inline vuint64_t
vatomic64_max_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_max_get(a, v);
}
static inline vuint64_t
vatomic64_and_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_and_get(a, v);
}
static inline vuint64_t
vatomic64_or_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_or_get(a, v);
}
static inline vuint64_t
vatomic64_xor_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xor_get(a, v);
}
static inline vuint64_t
vatomic64_add_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_add_get(a, v);
}
static inline vuint64_t
vatomic64_sub_get_rlx(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_sub_get(a, v);
}
static inline vuint64_t
vatomic64_get_max_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_max(a, v);
}
static inline vuint64_t
vatomic64_get_and_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_and(a, v);
}
static inline vuint64_t
vatomic64_get_or_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_or(a, v);
}
static inline vuint64_t
vatomic64_get_xor_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_xor(a, v);
}
static inline vuint64_t
vatomic64_get_add_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_add(a, v);
}
static inline vuint64_t
vatomic64_get_sub_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_sub(a, v);
}
static inline vuint64_t
vatomic64_max_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_max_get(a, v);
}
static inline vuint64_t
vatomic64_and_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_and_get(a, v);
}
static inline vuint64_t
vatomic64_or_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_or_get(a, v);
}
static inline vuint64_t
vatomic64_xor_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xor_get(a, v);
}
static inline vuint64_t
vatomic64_add_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_add_get(a, v);
}
static inline vuint64_t
vatomic64_sub_get_acq(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_sub_get(a, v);
}
static inline vuint64_t
vatomic64_get_max_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_max(a, v);
}
static inline vuint64_t
vatomic64_get_and_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_and(a, v);
}
static inline vuint64_t
vatomic64_get_or_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_or(a, v);
}
static inline vuint64_t
vatomic64_get_xor_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_xor(a, v);
}
static inline vuint64_t
vatomic64_get_add_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_add(a, v);
}
static inline vuint64_t
vatomic64_get_sub_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_sub(a, v);
}
static inline vuint64_t
vatomic64_max_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_max_get(a, v);
}
static inline vuint64_t
vatomic64_and_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_and_get(a, v);
}
static inline vuint64_t
vatomic64_or_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_or_get(a, v);
}
static inline vuint64_t
vatomic64_xor_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_xor_get(a, v);
}
static inline vuint64_t
vatomic64_add_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_add_get(a, v);
}
static inline vuint64_t
vatomic64_sub_get_rel(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_sub_get(a, v);
}
static inline vuint64_t
vatomic64_get_inc_rlx(vatomic64_t *a)
{
    return vatomic64_get_inc(a);
}
static inline vuint64_t
vatomic64_inc_get_rlx(vatomic64_t *a)
{
    return vatomic64_inc_get(a);
}
static inline vuint64_t
vatomic64_get_dec_rlx(vatomic64_t *a)
{
    return vatomic64_get_dec(a);
}
static inline vuint64_t
vatomic64_dec_get_rlx(vatomic64_t *a)
{
    return vatomic64_dec_get(a);
}
static inline vuint64_t
vatomic64_get_inc_acq(vatomic64_t *a)
{
    return vatomic64_get_inc(a);
}
static inline vuint64_t
vatomic64_inc_get_acq(vatomic64_t *a)
{
    return vatomic64_inc_get(a);
}
static inline vuint64_t
vatomic64_get_dec_acq(vatomic64_t *a)
{
    return vatomic64_get_dec(a);
}
static inline vuint64_t
vatomic64_dec_get_acq(vatomic64_t *a)
{
    return vatomic64_dec_get(a);
}
static inline vuint64_t
vatomic64_get_inc_rel(vatomic64_t *a)
{
    return vatomic64_get_inc(a);
}
static inline vuint64_t
vatomic64_inc_get_rel(vatomic64_t *a)
{
    return vatomic64_inc_get(a);
}
static inline vuint64_t
vatomic64_get_dec_rel(vatomic64_t *a)
{
    return vatomic64_get_dec(a);
}
static inline vuint64_t
vatomic64_dec_get_rel(vatomic64_t *a)
{
    return vatomic64_dec_get(a);
}
static inline void
vatomic64_max_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_max(a, v);
}
static inline void
vatomic64_and_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_and(a, v);
}
static inline void
vatomic64_or_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_or(a, v);
}
static inline void
vatomic64_xor_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_xor(a, v);
}
static inline void
vatomic64_add_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_add(a, v);
}
static inline void
vatomic64_sub_rlx(vatomic64_t *a, vuint64_t v)
{
    vatomic64_sub(a, v);
}
static inline void
vatomic64_max_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_max(a, v);
}
static inline void
vatomic64_and_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_and(a, v);
}
static inline void
vatomic64_or_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_or(a, v);
}
static inline void
vatomic64_xor_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_xor(a, v);
}
static inline void
vatomic64_add_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_add(a, v);
}
static inline void
vatomic64_sub_rel(vatomic64_t *a, vuint64_t v)
{
    vatomic64_sub(a, v);
}
static inline void
vatomic64_inc_rlx(vatomic64_t *a)
{
    vatomic64_inc(a);
}
static inline void
vatomic64_dec_rlx(vatomic64_t *a)
{
    vatomic64_dec(a);
}
static inline void
vatomic64_inc_rel(vatomic64_t *a)
{
    vatomic64_inc(a);
}
static inline void
vatomic64_dec_rel(vatomic64_t *a)
{
    vatomic64_dec(a);
}
static inline vuint64_t
vatomic64_await_eq_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_eq(a, v);
}
static inline vuint64_t
vatomic64_await_neq_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_neq(a, v);
}
static inline vuint64_t
vatomic64_await_lt_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_lt(a, v);
}
static inline vuint64_t
vatomic64_await_le_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_le(a, v);
}
static inline vuint64_t
vatomic64_await_gt_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_gt(a, v);
}
static inline vuint64_t
vatomic64_await_ge_rlx(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_ge(a, v);
}
static inline vuint64_t
vatomic64_await_eq_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_eq(a, v);
}
static inline vuint64_t
vatomic64_await_neq_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_neq(a, v);
}
static inline vuint64_t
vatomic64_await_lt_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_lt(a, v);
}
static inline vuint64_t
vatomic64_await_le_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_le(a, v);
}
static inline vuint64_t
vatomic64_await_gt_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_gt(a, v);
}
static inline vuint64_t
vatomic64_await_ge_acq(const vatomic64_t *a, vuint64_t v)
{
    return vatomic64_await_ge(a, v);
}
static inline vuint64_t
vatomic64_await_le_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_le_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_le_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_lt_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_lt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_ge_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_ge_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_gt_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_gt_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_neq_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_neq_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_add_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_add_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_add_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_add(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_sub_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_sub_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_sub_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_sub(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_set_rlx(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_set_acq(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_set(a, c, v);
}
static inline vuint64_t
vatomic64_await_eq_set_rel(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    return vatomic64_await_eq_set(a, c, v);
}
static inline vsize_t
vatomicsz_read_rlx(const vatomicsz_t *a)
{
    return vatomicsz_read(a);
}
static inline vsize_t
vatomicsz_read_acq(const vatomicsz_t *a)
{
    return vatomicsz_read(a);
}
static inline void
vatomicsz_write_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_write(a, v);
}
static inline void
vatomicsz_write_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_write(a, v);
}
static inline vsize_t
vatomicsz_xchg_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xchg(a, v);
}
static inline vsize_t
vatomicsz_xchg_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xchg(a, v);
}
static inline vsize_t
vatomicsz_xchg_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xchg(a, v);
}
static inline vsize_t
vatomicsz_cmpxchg_rlx(vatomicsz_t *a, vsize_t e, vsize_t v)
{
    return vatomicsz_cmpxchg(a, e, v);
}
static inline vsize_t
vatomicsz_cmpxchg_acq(vatomicsz_t *a, vsize_t e, vsize_t v)
{
    return vatomicsz_cmpxchg(a, e, v);
}
static inline vsize_t
vatomicsz_cmpxchg_rel(vatomicsz_t *a, vsize_t e, vsize_t v)
{
    return vatomicsz_cmpxchg(a, e, v);
}
static inline vsize_t
vatomicsz_get_max_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_max(a, v);
}
static inline vsize_t
vatomicsz_get_and_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_and(a, v);
}
static inline vsize_t
vatomicsz_get_or_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_or(a, v);
}
static inline vsize_t
vatomicsz_get_xor_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_xor(a, v);
}
static inline vsize_t
vatomicsz_get_add_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_add(a, v);
}
static inline vsize_t
vatomicsz_get_sub_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_sub(a, v);
}
static inline vsize_t
vatomicsz_max_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_max_get(a, v);
}
static inline vsize_t
vatomicsz_and_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_and_get(a, v);
}
static inline vsize_t
vatomicsz_or_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_or_get(a, v);
}
static inline vsize_t
vatomicsz_xor_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xor_get(a, v);
}
static inline vsize_t
vatomicsz_add_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_add_get(a, v);
}
static inline vsize_t
vatomicsz_sub_get_rlx(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_sub_get(a, v);
}
static inline vsize_t
vatomicsz_get_max_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_max(a, v);
}
static inline vsize_t
vatomicsz_get_and_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_and(a, v);
}
static inline vsize_t
vatomicsz_get_or_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_or(a, v);
}
static inline vsize_t
vatomicsz_get_xor_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_xor(a, v);
}
static inline vsize_t
vatomicsz_get_add_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_add(a, v);
}
static inline vsize_t
vatomicsz_get_sub_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_sub(a, v);
}
static inline vsize_t
vatomicsz_max_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_max_get(a, v);
}
static inline vsize_t
vatomicsz_and_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_and_get(a, v);
}
static inline vsize_t
vatomicsz_or_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_or_get(a, v);
}
static inline vsize_t
vatomicsz_xor_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xor_get(a, v);
}
static inline vsize_t
vatomicsz_add_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_add_get(a, v);
}
static inline vsize_t
vatomicsz_sub_get_acq(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_sub_get(a, v);
}
static inline vsize_t
vatomicsz_get_max_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_max(a, v);
}
static inline vsize_t
vatomicsz_get_and_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_and(a, v);
}
static inline vsize_t
vatomicsz_get_or_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_or(a, v);
}
static inline vsize_t
vatomicsz_get_xor_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_xor(a, v);
}
static inline vsize_t
vatomicsz_get_add_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_add(a, v);
}
static inline vsize_t
vatomicsz_get_sub_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_sub(a, v);
}
static inline vsize_t
vatomicsz_max_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_max_get(a, v);
}
static inline vsize_t
vatomicsz_and_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_and_get(a, v);
}
static inline vsize_t
vatomicsz_or_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_or_get(a, v);
}
static inline vsize_t
vatomicsz_xor_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_xor_get(a, v);
}
static inline vsize_t
vatomicsz_add_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_add_get(a, v);
}
static inline vsize_t
vatomicsz_sub_get_rel(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_sub_get(a, v);
}
static inline vsize_t
vatomicsz_get_inc_rlx(vatomicsz_t *a)
{
    return vatomicsz_get_inc(a);
}
static inline vsize_t
vatomicsz_inc_get_rlx(vatomicsz_t *a)
{
    return vatomicsz_inc_get(a);
}
static inline vsize_t
vatomicsz_get_dec_rlx(vatomicsz_t *a)
{
    return vatomicsz_get_dec(a);
}
static inline vsize_t
vatomicsz_dec_get_rlx(vatomicsz_t *a)
{
    return vatomicsz_dec_get(a);
}
static inline vsize_t
vatomicsz_get_inc_acq(vatomicsz_t *a)
{
    return vatomicsz_get_inc(a);
}
static inline vsize_t
vatomicsz_inc_get_acq(vatomicsz_t *a)
{
    return vatomicsz_inc_get(a);
}
static inline vsize_t
vatomicsz_get_dec_acq(vatomicsz_t *a)
{
    return vatomicsz_get_dec(a);
}
static inline vsize_t
vatomicsz_dec_get_acq(vatomicsz_t *a)
{
    return vatomicsz_dec_get(a);
}
static inline vsize_t
vatomicsz_get_inc_rel(vatomicsz_t *a)
{
    return vatomicsz_get_inc(a);
}
static inline vsize_t
vatomicsz_inc_get_rel(vatomicsz_t *a)
{
    return vatomicsz_inc_get(a);
}
static inline vsize_t
vatomicsz_get_dec_rel(vatomicsz_t *a)
{
    return vatomicsz_get_dec(a);
}
static inline vsize_t
vatomicsz_dec_get_rel(vatomicsz_t *a)
{
    return vatomicsz_dec_get(a);
}
static inline void
vatomicsz_max_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_max(a, v);
}
static inline void
vatomicsz_and_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_and(a, v);
}
static inline void
vatomicsz_or_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_or(a, v);
}
static inline void
vatomicsz_xor_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_xor(a, v);
}
static inline void
vatomicsz_add_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_add(a, v);
}
static inline void
vatomicsz_sub_rlx(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_sub(a, v);
}
static inline void
vatomicsz_max_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_max(a, v);
}
static inline void
vatomicsz_and_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_and(a, v);
}
static inline void
vatomicsz_or_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_or(a, v);
}
static inline void
vatomicsz_xor_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_xor(a, v);
}
static inline void
vatomicsz_add_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_add(a, v);
}
static inline void
vatomicsz_sub_rel(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_sub(a, v);
}
static inline void
vatomicsz_inc_rlx(vatomicsz_t *a)
{
    vatomicsz_inc(a);
}
static inline void
vatomicsz_dec_rlx(vatomicsz_t *a)
{
    vatomicsz_dec(a);
}
static inline void
vatomicsz_inc_rel(vatomicsz_t *a)
{
    vatomicsz_inc(a);
}
static inline void
vatomicsz_dec_rel(vatomicsz_t *a)
{
    vatomicsz_dec(a);
}
static inline void *
vatomicptr_read_rlx(const vatomicptr_t *a)
{
    return vatomicptr_read(a);
}
static inline void *
vatomicptr_read_acq(const vatomicptr_t *a)
{
    return vatomicptr_read(a);
}
static inline void
vatomicptr_write_rlx(vatomicptr_t *a, void *v)
{
    vatomicptr_write(a, v);
}
static inline void
vatomicptr_write_rel(vatomicptr_t *a, void *v)
{
    vatomicptr_write(a, v);
}
static inline void *
vatomicptr_xchg_rlx(vatomicptr_t *a, void *v)
{
    return vatomicptr_xchg(a, v);
}
static inline void *
vatomicptr_xchg_acq(vatomicptr_t *a, void *v)
{
    return vatomicptr_xchg(a, v);
}
static inline void *
vatomicptr_xchg_rel(vatomicptr_t *a, void *v)
{
    return vatomicptr_xchg(a, v);
}
static inline void *
vatomicptr_cmpxchg_rlx(vatomicptr_t *a, void *e, void *v)
{
    return vatomicptr_cmpxchg(a, e, v);
}
static inline void *
vatomicptr_cmpxchg_acq(vatomicptr_t *a, void *e, void *v)
{
    return vatomicptr_cmpxchg(a, e, v);
}
static inline void *
vatomicptr_cmpxchg_rel(vatomicptr_t *a, void *e, void *v)
{
    return vatomicptr_cmpxchg(a, e, v);
}
static inline void *
vatomicptr_await_eq_rlx(const vatomicptr_t *a, void *v)
{
    return vatomicptr_await_eq(a, v);
}
static inline void *
vatomicptr_await_neq_rlx(const vatomicptr_t *a, void *v)
{
    return vatomicptr_await_neq(a, v);
}
static inline void *
vatomicptr_await_eq_acq(const vatomicptr_t *a, void *v)
{
    return vatomicptr_await_eq(a, v);
}
static inline void *
vatomicptr_await_neq_acq(const vatomicptr_t *a, void *v)
{
    return vatomicptr_await_neq(a, v);
}
static inline void *
vatomicptr_await_neq_set_rlx(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_neq_set(a, c, v);
}
static inline void *
vatomicptr_await_neq_set_acq(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_neq_set(a, c, v);
}
static inline void *
vatomicptr_await_neq_set_rel(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_neq_set(a, c, v);
}
static inline void *
vatomicptr_await_eq_set_rlx(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_eq_set(a, c, v);
}
static inline void *
vatomicptr_await_eq_set_acq(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_eq_set(a, c, v);
}
static inline void *
vatomicptr_await_eq_set_rel(vatomicptr_t *a, void *c, void *v)
{
    return vatomicptr_await_eq_set(a, c, v);
}
static inline void
vatomic_fence(void)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_thread_fence(5);
    __asm__ __volatile__("" ::: "memory");
}
static inline vuint8_t
vatomic8_read(const vatomic8_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp = (vuint8_t)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_read(const vatomic16_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp = (vuint16_t)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_read(const vatomic32_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp = (vuint32_t)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_read(const vatomic64_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp = (vuint64_t)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_read(const vatomicsz_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp = (vsize_t)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline void *
vatomicptr_read(const vatomicptr_t *a)
{
    __asm__ __volatile__("" ::: "memory");
    void *tmp = (void *)__atomic_load_n(&a->_v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline void
vatomic8_write(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline void
vatomic16_write(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline void
vatomic32_write(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline void
vatomic64_write(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline void
vatomicsz_write(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline void
vatomicptr_write(vatomicptr_t *a, void *v)
{
    __asm__ __volatile__("" ::: "memory");
    __atomic_store_n(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
}
static inline vuint8_t
vatomic8_xchg(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp =
        (vuint8_t)__atomic_exchange_n(&a->_v, (vuint8_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_xchg(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp =
        (vuint16_t)__atomic_exchange_n(&a->_v, (vuint16_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_xchg(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp =
        (vuint32_t)__atomic_exchange_n(&a->_v, (vuint32_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_xchg(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp =
        (vuint64_t)__atomic_exchange_n(&a->_v, (vuint64_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_xchg(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp =
        (vsize_t)__atomic_exchange_n(&a->_v, (vsize_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline void *
vatomicptr_xchg(vatomicptr_t *a, void *v)
{
    __asm__ __volatile__("" ::: "memory");
    void *tmp =
        (void *)__atomic_exchange_n(&a->_v, (void *)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_cmpxchg(vatomic8_t *a, vuint8_t e, vuint8_t v)
{
    vuint8_t exp = (vuint8_t)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (vuint8_t)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline vuint16_t
vatomic16_cmpxchg(vatomic16_t *a, vuint16_t e, vuint16_t v)
{
    vuint16_t exp = (vuint16_t)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (vuint16_t)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline vuint32_t
vatomic32_cmpxchg(vatomic32_t *a, vuint32_t e, vuint32_t v)
{
    vuint32_t exp = (vuint32_t)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (vuint32_t)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline vuint64_t
vatomic64_cmpxchg(vatomic64_t *a, vuint64_t e, vuint64_t v)
{
    vuint64_t exp = (vuint64_t)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (vuint64_t)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline vsize_t
vatomicsz_cmpxchg(vatomicsz_t *a, vsize_t e, vsize_t v)
{
    vsize_t exp = (vsize_t)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (vsize_t)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline void *
vatomicptr_cmpxchg(vatomicptr_t *a, void *e, void *v)
{
    void *exp = (void *)e;
    __asm__ __volatile__("" ::: "memory");
    __atomic_compare_exchange_n(&a->_v, &exp, (void *)v, 0, 5,
                                5);
    __asm__ __volatile__("" ::: "memory");
    return exp;
}
static inline vuint8_t
vatomic8_get_and(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp =
        (vuint8_t)__atomic_fetch_and(&a->_v, (vuint8_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_get_and(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp =
        (vuint16_t)__atomic_fetch_and(&a->_v, (vuint16_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_get_and(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp =
        (vuint32_t)__atomic_fetch_and(&a->_v, (vuint32_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_get_and(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp =
        (vuint64_t)__atomic_fetch_and(&a->_v, (vuint64_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_get_and(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp =
        (vsize_t)__atomic_fetch_and(&a->_v, (vsize_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_get_or(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp =
        (vuint8_t)__atomic_fetch_or(&a->_v, (vuint8_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_get_or(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp =
        (vuint16_t)__atomic_fetch_or(&a->_v, (vuint16_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_get_or(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp =
        (vuint32_t)__atomic_fetch_or(&a->_v, (vuint32_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_get_or(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp =
        (vuint64_t)__atomic_fetch_or(&a->_v, (vuint64_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_get_or(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp =
        (vsize_t)__atomic_fetch_or(&a->_v, (vsize_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_get_xor(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp =
        (vuint8_t)__atomic_fetch_xor(&a->_v, (vuint8_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_get_xor(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp =
        (vuint16_t)__atomic_fetch_xor(&a->_v, (vuint16_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_get_xor(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp =
        (vuint32_t)__atomic_fetch_xor(&a->_v, (vuint32_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_get_xor(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp =
        (vuint64_t)__atomic_fetch_xor(&a->_v, (vuint64_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_get_xor(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp =
        (vsize_t)__atomic_fetch_xor(&a->_v, (vsize_t)v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_get_add(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp = (vuint8_t)__atomic_fetch_add(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_get_add(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp = (vuint16_t)__atomic_fetch_add(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_get_add(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp = (vuint32_t)__atomic_fetch_add(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_get_add(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp = (vuint64_t)__atomic_fetch_add(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_get_add(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp = (vsize_t)__atomic_fetch_add(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_get_sub(vatomic8_t *a, vuint8_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint8_t tmp = __atomic_fetch_sub(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint16_t
vatomic16_get_sub(vatomic16_t *a, vuint16_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint16_t tmp = __atomic_fetch_sub(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint32_t
vatomic32_get_sub(vatomic32_t *a, vuint32_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint32_t tmp = __atomic_fetch_sub(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint64_t
vatomic64_get_sub(vatomic64_t *a, vuint64_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vuint64_t tmp = __atomic_fetch_sub(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vsize_t
vatomicsz_get_sub(vatomicsz_t *a, vsize_t v)
{
    __asm__ __volatile__("" ::: "memory");
    vsize_t tmp = __atomic_fetch_sub(&a->_v, v, 5);
    __asm__ __volatile__("" ::: "memory");
    return tmp;
}
static inline vuint8_t
vatomic8_get_max(vatomic8_t *a, vuint8_t v)
{
    vuint8_t old = 0;
    vuint8_t cur = vatomic8_read(a);
    do {
        old = cur;
        if (old >= v) {
            break;
        }
        cur = vatomic8_cmpxchg(a, old, v);
    } while (cur != old);
    return old;
}
static inline vuint16_t
vatomic16_get_max(vatomic16_t *a, vuint16_t v)
{
    vuint16_t old = 0;
    vuint16_t cur = vatomic16_read(a);
    do {
        old = cur;
        if (old >= v) {
            break;
        }
        cur = vatomic16_cmpxchg(a, old, v);
    } while (cur != old);
    return old;
}
static inline vuint32_t
vatomic32_get_max(vatomic32_t *a, vuint32_t v)
{
    vuint32_t old = 0;
    vuint32_t cur = vatomic32_read(a);
    do {
        old = cur;
        if (old >= v) {
            break;
        }
        cur = vatomic32_cmpxchg(a, old, v);
    } while (cur != old);
    return old;
}
static inline vuint64_t
vatomic64_get_max(vatomic64_t *a, vuint64_t v)
{
    vuint64_t old = 0;
    vuint64_t cur = vatomic64_read(a);
    do {
        old = cur;
        if (old >= v) {
            break;
        }
        cur = vatomic64_cmpxchg(a, old, v);
    } while (cur != old);
    return old;
}
static inline vsize_t
vatomicsz_get_max(vatomicsz_t *a, vsize_t v)
{
    vsize_t old = 0;
    vsize_t cur = vatomicsz_read(a);
    do {
        old = cur;
        if (old >= v) {
            break;
        }
        cur = vatomicsz_cmpxchg(a, old, v);
    } while (cur != old);
    return old;
}
static inline vuint8_t
vatomic8_max_get(vatomic8_t *a, vuint8_t v)
{
    vuint8_t o = vatomic8_get_max(a, v);
    return o >= v ? o : v;
}
static inline vuint16_t
vatomic16_max_get(vatomic16_t *a, vuint16_t v)
{
    vuint16_t o = vatomic16_get_max(a, v);
    return o >= v ? o : v;
}
static inline vuint32_t
vatomic32_max_get(vatomic32_t *a, vuint32_t v)
{
    vuint32_t o = vatomic32_get_max(a, v);
    return o >= v ? o : v;
}
static inline vuint64_t
vatomic64_max_get(vatomic64_t *a, vuint64_t v)
{
    vuint64_t o = vatomic64_get_max(a, v);
    return o >= v ? o : v;
}
static inline vsize_t
vatomicsz_max_get(vatomicsz_t *a, vsize_t v)
{
    vsize_t o = vatomicsz_get_max(a, v);
    return o >= v ? o : v;
}
static inline void
vatomic8_max(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_max(a, v);
}
static inline void
vatomic16_max(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_max(a, v);
}
static inline void
vatomic32_max(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_max(a, v);
}
static inline void
vatomic64_max(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_max(a, v);
}
static inline void
vatomicsz_max(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_max(a, v);
}
static inline vuint8_t
vatomic8_and_get(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_and(a, v) & v;
}
static inline vuint16_t
vatomic16_and_get(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_and(a, v) & v;
}
static inline vuint32_t
vatomic32_and_get(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_and(a, v) & v;
}
static inline vuint64_t
vatomic64_and_get(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_and(a, v) & v;
}
static inline vsize_t
vatomicsz_and_get(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_and(a, v) & v;
}
static inline void
vatomic8_and(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_and(a, v);
}
static inline void
vatomic16_and(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_and(a, v);
}
static inline void
vatomic32_and(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_and(a, v);
}
static inline void
vatomic64_and(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_and(a, v);
}
static inline void
vatomicsz_and(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_and(a, v);
}
static inline vuint8_t
vatomic8_or_get(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_or(a, v) | v;
}
static inline vuint16_t
vatomic16_or_get(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_or(a, v) | v;
}
static inline vuint32_t
vatomic32_or_get(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_or(a, v) | v;
}
static inline vuint64_t
vatomic64_or_get(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_or(a, v) | v;
}
static inline vsize_t
vatomicsz_or_get(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_or(a, v) | v;
}
static inline void
vatomic8_or(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_or(a, v);
}
static inline void
vatomic16_or(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_or(a, v);
}
static inline void
vatomic32_or(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_or(a, v);
}
static inline void
vatomic64_or(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_or(a, v);
}
static inline void
vatomicsz_or(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_or(a, v);
}
static inline vuint8_t
vatomic8_xor_get(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_xor(a, v) ^ v;
}
static inline vuint16_t
vatomic16_xor_get(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_xor(a, v) ^ v;
}
static inline vuint32_t
vatomic32_xor_get(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_xor(a, v) ^ v;
}
static inline vuint64_t
vatomic64_xor_get(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_xor(a, v) ^ v;
}
static inline vsize_t
vatomicsz_xor_get(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_xor(a, v) ^ v;
}
static inline void
vatomic8_xor(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_xor(a, v);
}
static inline void
vatomic16_xor(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_xor(a, v);
}
static inline void
vatomic32_xor(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_xor(a, v);
}
static inline void
vatomic64_xor(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_xor(a, v);
}
static inline void
vatomicsz_xor(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_xor(a, v);
}
static inline vuint8_t
vatomic8_add_get(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_add(a, v) + v;
}
static inline vuint16_t
vatomic16_add_get(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_add(a, v) + v;
}
static inline vuint32_t
vatomic32_add_get(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_add(a, v) + v;
}
static inline vuint64_t
vatomic64_add_get(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_add(a, v) + v;
}
static inline vsize_t
vatomicsz_add_get(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_add(a, v) + v;
}
static inline void
vatomic8_add(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_add(a, v);
}
static inline void
vatomic16_add(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_add(a, v);
}
static inline void
vatomic32_add(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_add(a, v);
}
static inline void
vatomic64_add(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_add(a, v);
}
static inline void
vatomicsz_add(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_add(a, v);
}
static inline vuint8_t
vatomic8_get_inc(vatomic8_t *a)
{
    return vatomic8_get_add(a, 1U);
}
static inline vuint16_t
vatomic16_get_inc(vatomic16_t *a)
{
    return vatomic16_get_add(a, 1U);
}
static inline vuint32_t
vatomic32_get_inc(vatomic32_t *a)
{
    return vatomic32_get_add(a, 1U);
}
static inline vuint64_t
vatomic64_get_inc(vatomic64_t *a)
{
    return vatomic64_get_add(a, 1U);
}
static inline vsize_t
vatomicsz_get_inc(vatomicsz_t *a)
{
    return vatomicsz_get_add(a, 1U);
}
static inline vuint8_t
vatomic8_inc_get(vatomic8_t *a)
{
    return vatomic8_add_get(a, 1U);
}
static inline vuint16_t
vatomic16_inc_get(vatomic16_t *a)
{
    return vatomic16_add_get(a, 1U);
}
static inline vuint32_t
vatomic32_inc_get(vatomic32_t *a)
{
    return vatomic32_add_get(a, 1U);
}
static inline vuint64_t
vatomic64_inc_get(vatomic64_t *a)
{
    return vatomic64_add_get(a, 1U);
}
static inline vsize_t
vatomicsz_inc_get(vatomicsz_t *a)
{
    return vatomicsz_add_get(a, 1U);
}
static inline void
vatomic8_inc(vatomic8_t *a)
{
    (void)vatomic8_get_inc(a);
}
static inline void
vatomic8_inc_acq(vatomic8_t *a)
{
    (void)vatomic8_get_inc_acq(a);
}
static inline void
vatomic16_inc(vatomic16_t *a)
{
    (void)vatomic16_get_inc(a);
}
static inline void
vatomic16_inc_acq(vatomic16_t *a)
{
    (void)vatomic16_get_inc_acq(a);
}
static inline void
vatomic32_inc(vatomic32_t *a)
{
    (void)vatomic32_get_inc(a);
}
static inline void
vatomic32_inc_acq(vatomic32_t *a)
{
    (void)vatomic32_get_inc_acq(a);
}
static inline void
vatomic64_inc(vatomic64_t *a)
{
    (void)vatomic64_get_inc(a);
}
static inline void
vatomic64_inc_acq(vatomic64_t *a)
{
    (void)vatomic64_get_inc_acq(a);
}
static inline void
vatomicsz_inc(vatomicsz_t *a)
{
    (void)vatomicsz_get_inc(a);
}
static inline void
vatomicsz_inc_acq(vatomicsz_t *a)
{
    (void)vatomicsz_get_inc_acq(a);
}
static inline vuint8_t
vatomic8_sub_get(vatomic8_t *a, vuint8_t v)
{
    return vatomic8_get_sub(a, v) - v;
}
static inline vuint16_t
vatomic16_sub_get(vatomic16_t *a, vuint16_t v)
{
    return vatomic16_get_sub(a, v) - v;
}
static inline vuint32_t
vatomic32_sub_get(vatomic32_t *a, vuint32_t v)
{
    return vatomic32_get_sub(a, v) - v;
}
static inline vuint64_t
vatomic64_sub_get(vatomic64_t *a, vuint64_t v)
{
    return vatomic64_get_sub(a, v) - v;
}
static inline vsize_t
vatomicsz_sub_get(vatomicsz_t *a, vsize_t v)
{
    return vatomicsz_get_sub(a, v) - v;
}
static inline void
vatomic8_sub(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_sub(a, v);
}
static inline void
vatomic8_sub_acq(vatomic8_t *a, vuint8_t v)
{
    (void)vatomic8_get_sub_acq(a, v);
}
static inline void
vatomic16_sub(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_sub(a, v);
}
static inline void
vatomic16_sub_acq(vatomic16_t *a, vuint16_t v)
{
    (void)vatomic16_get_sub_acq(a, v);
}
static inline void
vatomic32_sub(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_sub(a, v);
}
static inline void
vatomic32_sub_acq(vatomic32_t *a, vuint32_t v)
{
    (void)vatomic32_get_sub_acq(a, v);
}
static inline void
vatomic64_sub(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_sub(a, v);
}
static inline void
vatomic64_sub_acq(vatomic64_t *a, vuint64_t v)
{
    (void)vatomic64_get_sub_acq(a, v);
}
static inline void
vatomicsz_sub(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_sub(a, v);
}
static inline void
vatomicsz_sub_acq(vatomicsz_t *a, vsize_t v)
{
    (void)vatomicsz_get_sub_acq(a, v);
}
static inline vuint8_t
vatomic8_get_dec(vatomic8_t *a)
{
    return vatomic8_get_sub(a, 1U);
}
static inline vuint16_t
vatomic16_get_dec(vatomic16_t *a)
{
    return vatomic16_get_sub(a, 1U);
}
static inline vuint32_t
vatomic32_get_dec(vatomic32_t *a)
{
    return vatomic32_get_sub(a, 1U);
}
static inline vuint64_t
vatomic64_get_dec(vatomic64_t *a)
{
    return vatomic64_get_sub(a, 1U);
}
static inline vsize_t
vatomicsz_get_dec(vatomicsz_t *a)
{
    return vatomicsz_get_sub(a, 1U);
}
static inline vuint8_t
vatomic8_dec_get(vatomic8_t *a)
{
    return vatomic8_sub_get(a, 1U);
}
static inline vuint16_t
vatomic16_dec_get(vatomic16_t *a)
{
    return vatomic16_sub_get(a, 1U);
}
static inline vuint32_t
vatomic32_dec_get(vatomic32_t *a)
{
    return vatomic32_sub_get(a, 1U);
}
static inline vuint64_t
vatomic64_dec_get(vatomic64_t *a)
{
    return vatomic64_sub_get(a, 1U);
}
static inline vsize_t
vatomicsz_dec_get(vatomicsz_t *a)
{
    return vatomicsz_sub_get(a, 1U);
}
static inline void
vatomic8_dec(vatomic8_t *a)
{
    (void)vatomic8_get_dec(a);
}
static inline void
vatomic8_dec_acq(vatomic8_t *a)
{
    (void)vatomic8_get_dec_acq(a);
}
static inline void
vatomic16_dec(vatomic16_t *a)
{
    (void)vatomic16_get_dec(a);
}
static inline void
vatomic16_dec_acq(vatomic16_t *a)
{
    (void)vatomic16_get_dec_acq(a);
}
static inline void
vatomic32_dec(vatomic32_t *a)
{
    (void)vatomic32_get_dec(a);
}
static inline void
vatomic32_dec_acq(vatomic32_t *a)
{
    (void)vatomic32_get_dec_acq(a);
}
static inline void
vatomic64_dec(vatomic64_t *a)
{
    (void)vatomic64_get_dec(a);
}
static inline void
vatomic64_dec_acq(vatomic64_t *a)
{
    (void)vatomic64_get_dec_acq(a);
}
static inline void
vatomicsz_dec(vatomicsz_t *a)
{
    (void)vatomicsz_get_dec(a);
}
static inline void
vatomicsz_dec_acq(vatomicsz_t *a)
{
    (void)vatomicsz_get_dec_acq(a);
}
static inline void
vatomic8_init(vatomic8_t *a, vuint8_t v)
{
    vatomic8_write(a, v);
}
static inline void
vatomic16_init(vatomic16_t *a, vuint16_t v)
{
    vatomic16_write(a, v);
}
static inline void
vatomic32_init(vatomic32_t *a, vuint32_t v)
{
    vatomic32_write(a, v);
}
static inline void
vatomic64_init(vatomic64_t *a, vuint64_t v)
{
    vatomic64_write(a, v);
}
static inline void
vatomicsz_init(vatomicsz_t *a, vsize_t v)
{
    vatomicsz_write(a, v);
}
static inline void
vatomicptr_init(vatomicptr_t *a, void *v)
{
    vatomicptr_write(a, v);
}
static inline vuint32_t
vatomic32_await_neq(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t cur = 0;
    for (verification_loop_begin(); (verification_spin_start(), ((cur = vatomic32_read(a), cur == c)) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
    }
    return cur;
}
static inline vuint64_t
vatomic64_await_neq(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t cur = 0;
    for (verification_loop_begin(); (verification_spin_start(), ((cur = vatomic64_read(a), cur == c)) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
    }
    return cur;
}
static inline void *
vatomicptr_await_neq(const vatomicptr_t *a, void *c)
{
    void *cur = ((void *)0);
    for (verification_loop_begin(); (verification_spin_start(), ((cur = vatomicptr_read(a), cur == c)) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
    }
    return cur;
}
static inline vuint32_t
vatomic32_await_eq(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t ret = c;
    vuint32_t o = 0;
    for (verification_loop_begin(); (verification_spin_start(), ((o = vatomic32_read(a)) != c) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
        ret = o;
    }
    return ret;
}
static inline vuint64_t
vatomic64_await_eq(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t ret = c;
    vuint64_t o = 0;
    for (verification_loop_begin(); (verification_spin_start(), ((o = vatomic64_read(a)) != c) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
        ret = o;
    }
    return ret;
}
static inline void *
vatomicptr_await_eq(const vatomicptr_t *a, void *c)
{
    void *ret = c;
    void *o = ((void *)0);
    for (verification_loop_begin(); (verification_spin_start(), ((o = vatomicptr_read(a)) != c) ? 1 : (verification_spin_end(1), 0)); verification_spin_end(0)) {
        do { } while (0);
        ret = o;
    }
    return ret;
}
static inline vuint32_t
vatomic32_await_le(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t old = vatomic32_read(a);
    while (!(old <= c)) {
        old = vatomic32_await_neq(a, old);
    }
    return old;
}
static inline vuint32_t
vatomic32_await_lt(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t old = vatomic32_read(a);
    while (!(old < c)) {
        old = vatomic32_await_neq(a, old);
    }
    return old;
}
static inline vuint32_t
vatomic32_await_ge(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t old = vatomic32_read(a);
    while (!(old >= c)) {
        old = vatomic32_await_neq(a, old);
    }
    return old;
}
static inline vuint32_t
vatomic32_await_gt(const vatomic32_t *a, vuint32_t c)
{
    vuint32_t old = vatomic32_read(a);
    while (!(old > c)) {
        old = vatomic32_await_neq(a, old);
    }
    return old;
}
static inline vuint64_t
vatomic64_await_le(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t old = vatomic64_read(a);
    while (!(old <= c)) {
        old = vatomic64_await_neq(a, old);
    }
    return old;
}
static inline vuint64_t
vatomic64_await_lt(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t old = vatomic64_read(a);
    while (!(old < c)) {
        old = vatomic64_await_neq(a, old);
    }
    return old;
}
static inline vuint64_t
vatomic64_await_ge(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t old = vatomic64_read(a);
    while (!(old >= c)) {
        old = vatomic64_await_neq(a, old);
    }
    return old;
}
static inline vuint64_t
vatomic64_await_gt(const vatomic64_t *a, vuint64_t c)
{
    vuint64_t old = vatomic64_read(a);
    while (!(old > c)) {
        old = vatomic64_await_neq(a, old);
    }
    return old;
}
static inline vuint32_t
vatomic32_await_le_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_le_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_le_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_lt_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_lt_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_lt_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_ge_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_ge_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_ge_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_gt_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_gt_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_gt_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t cur = 0;
    vuint32_t old = vatomic32_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic32_await_neq(a, cur);
        }
    } while ((old = vatomic32_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_le_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_le_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_le_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur <= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_lt_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_lt_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_lt_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur < c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_ge_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_ge_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_ge_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur >= c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_gt_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur + v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_gt_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, cur - v)) != cur);
    return old;
}
static inline vuint64_t
vatomic64_await_gt_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t cur = 0;
    vuint64_t old = vatomic64_read(a);
    do {
        cur = old;
        while (!(cur > c)) {
            cur = vatomic64_await_neq(a, cur);
        }
    } while ((old = vatomic64_cmpxchg(a, cur, v)) != cur);
    return old;
}
static inline vuint32_t
vatomic32_await_neq_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t old = 0;
    do {
        old = vatomic32_await_neq(a, c);
    } while (vatomic32_cmpxchg(a, old, old + v) != old);
    return old;
}
static inline vuint32_t
vatomic32_await_neq_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t old = 0;
    do {
        old = vatomic32_await_neq(a, c);
    } while (vatomic32_cmpxchg(a, old, old - v) != old);
    return old;
}
static inline vuint32_t
vatomic32_await_neq_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    vuint32_t old = 0;
    do {
        old = vatomic32_await_neq(a, c);
    } while (vatomic32_cmpxchg(a, old, v) != old);
    return old;
}
static inline vuint64_t
vatomic64_await_neq_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t old = 0;
    do {
        old = vatomic64_await_neq(a, c);
    } while (vatomic64_cmpxchg(a, old, old + v) != old);
    return old;
}
static inline vuint64_t
vatomic64_await_neq_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t old = 0;
    do {
        old = vatomic64_await_neq(a, c);
    } while (vatomic64_cmpxchg(a, old, old - v) != old);
    return old;
}
static inline vuint64_t
vatomic64_await_neq_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    vuint64_t old = 0;
    do {
        old = vatomic64_await_neq(a, c);
    } while (vatomic64_cmpxchg(a, old, v) != old);
    return old;
}
static inline void *
vatomicptr_await_neq_set(vatomicptr_t *a, void *c, void *v)
{
    void *old = ((void *)0);
    do {
        old = vatomicptr_await_neq(a, c);
    } while (vatomicptr_cmpxchg(a, old, v) != old);
    return old;
}
static inline vuint32_t
vatomic32_await_eq_add(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    do {
        (void)vatomic32_await_eq(a, c);
    } while (vatomic32_cmpxchg(a, c, c + v) != c);
    return c;
}
static inline vuint32_t
vatomic32_await_eq_sub(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    do {
        (void)vatomic32_await_eq(a, c);
    } while (vatomic32_cmpxchg(a, c, c - v) != c);
    return c;
}
static inline vuint32_t
vatomic32_await_eq_set(vatomic32_t *a, vuint32_t c, vuint32_t v)
{
    do {
        (void)vatomic32_await_eq(a, c);
    } while (vatomic32_cmpxchg(a, c, v) != c);
    return c;
}
static inline vuint64_t
vatomic64_await_eq_add(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    do {
        (void)vatomic64_await_eq(a, c);
    } while (vatomic64_cmpxchg(a, c, c + v) != c);
    return c;
}
static inline vuint64_t
vatomic64_await_eq_sub(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    do {
        (void)vatomic64_await_eq(a, c);
    } while (vatomic64_cmpxchg(a, c, c - v) != c);
    return c;
}
static inline vuint64_t
vatomic64_await_eq_set(vatomic64_t *a, vuint64_t c, vuint64_t v)
{
    do {
        (void)vatomic64_await_eq(a, c);
    } while (vatomic64_cmpxchg(a, c, v) != c);
    return c;
}
static inline void *
vatomicptr_await_eq_set(vatomicptr_t *a, void *c, void *v)
{
    do {
        (void)vatomicptr_await_eq(a, c);
    } while (vatomicptr_cmpxchg(a, c, v) != c);
    return c;
}
typedef enum {
    QUEUE_BOUNDED_OK = 0,
    QUEUE_BOUNDED_FULL = 1,
    QUEUE_BOUNDED_EMPTY = 2,
    QUEUE_BOUNDED_AGAIN = 3,
} bounded_ret_t;
typedef struct bounded_mpmc_s {
    vatomic32_t phead;
    vatomic32_t ptail;
    vatomic32_t chead;
    vatomic32_t ctail;
    void **buf;
    vuint32_t size;
} bounded_mpmc_t;
static inline void
bounded_mpmc_init(bounded_mpmc_t *q, void **b, vuint32_t s)
{
    do { if (!(b && "buffer is NULL")) reach_error(); } while (0);
    do { if (!(s != 0 && "buffer with 0 size")) reach_error(); } while (0);
    q->buf = b;
    q->size = s;
    vatomic32_init(&q->chead, 0);
    vatomic32_init(&q->ctail, 0);
    vatomic32_init(&q->phead, 0);
    vatomic32_init(&q->ptail, 0);
}
static inline bounded_ret_t
bounded_mpmc_enq(bounded_mpmc_t *q, void *v)
{
    vuint32_t curr, next;
    curr = vatomic32_read_acq(&q->phead);
    if (curr - vatomic32_read_rlx(&q->ctail) == q->size) {
        return QUEUE_BOUNDED_FULL;
    }
    next = curr + 1;
    if (vatomic32_cmpxchg_rel(&q->phead, curr, next) != curr) {
        return QUEUE_BOUNDED_AGAIN;
    }
    q->buf[curr % q->size] = v;
    vatomic32_write_rel(&q->ptail, next);
    return QUEUE_BOUNDED_OK;
}
static inline bounded_ret_t
bounded_mpmc_deq(bounded_mpmc_t *q, void **v)
{
    vuint32_t curr, next;
    curr = vatomic32_read_acq(&q->chead);
    next = curr + 1;
    if (curr == vatomic32_read_acq(&q->ptail)) {
        return QUEUE_BOUNDED_EMPTY;
    }
    if (vatomic32_cmpxchg_rel(&q->chead, curr, next) != curr) {
        return QUEUE_BOUNDED_AGAIN;
    }
    *v = q->buf[curr % q->size];
    vatomic32_await_eq_rlx(&q->ctail, curr);
    vatomic32_write_rel(&q->ctail, next);
    return QUEUE_BOUNDED_OK;
}
typedef int xbo_cb(void);
typedef struct xbo_s {
    vuint32_t min, max;
    vuint32_t factor;
    vuint32_t value;
} xbo_t;
static inline void
xbo_init(xbo_t *xbo, vuint32_t min, vuint32_t max, vuint32_t factor)
{
}
static inline void
xbo_backoff(xbo_t *xbo, xbo_cb *nop, xbo_cb *cb)
{
}
static inline void
xbo_reset(xbo_t *xbo)
{
}
static inline int
xbo_nop(void)
{
    volatile int k = 0;
    return k;
}
int
sched_yield(void)
{
    return 0;
}
void *g_buf[(2 * 2)];
bounded_mpmc_t g_queue;
vsize_t g_val[(2 * 2)];
void *
writer(void *arg)
{
    vsize_t tid = (vsize_t)(vuintptr_t)arg;
    xbo_t xbo;
    xbo_init(&xbo, 0, 100U, 2U);
    for (vsize_t i = 0; i < 2; i++) {
        vsize_t idx = tid * 2 + i;
        g_val[idx] = tid * 10U + i + 1;
        while (bounded_mpmc_enq(&g_queue, &g_val[idx]) != QUEUE_BOUNDED_OK) {
            xbo_backoff(&xbo, xbo_nop, sched_yield);
        }
        xbo_reset(&xbo);
    }
    return ((void *)0);
}
int
main(void)
{
    bounded_ret_t ret = QUEUE_BOUNDED_OK;
    bounded_mpmc_init(&g_queue, g_buf, (2 * 2));
    pthread_t t[2];
    for (vuintptr_t i = 0; i < 2; i++) {
        (void)pthread_create(&t[i], 0, writer, (void *)i);
    }
    for (vuintptr_t i = 0; i < 2; i++) {
        (void)pthread_join(t[i], ((void *)0));
    }
    void *r = ((void *)0);
    ret = bounded_mpmc_deq(&g_queue, &r);
    do { if (!(ret == QUEUE_BOUNDED_OK)) reach_error(); } while (0);
    vsize_t dequeued = *((vsize_t *)r);
    do { if (!((dequeued % 10U) == 1)) reach_error(); } while (0);
    do { if (!(dequeued <= (2 * 10U + 1))) reach_error(); } while (0);
    ret = bounded_mpmc_enq(&g_queue, &r);
    do { if (!(ret == QUEUE_BOUNDED_OK)) reach_error(); } while (0);
    ret = bounded_mpmc_enq(&g_queue, &r);
    do { if (!(ret == QUEUE_BOUNDED_FULL)) reach_error(); } while (0);
    do { do { (void)(ret); do { (void)(dequeued); do { } while (0); } while (0); } while (0); } while (0);
    return 0;
}
