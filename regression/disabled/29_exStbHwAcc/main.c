









typedef unsigned int size_t;



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







__extension__ typedef long long int __quad_t;
__extension__ typedef unsigned long long int __u_quad_t;


__extension__ typedef __u_quad_t __dev_t;
__extension__ typedef unsigned int __uid_t;
__extension__ typedef unsigned int __gid_t;
__extension__ typedef unsigned long int __ino_t;
__extension__ typedef __u_quad_t __ino64_t;
__extension__ typedef unsigned int __mode_t;
__extension__ typedef unsigned int __nlink_t;
__extension__ typedef long int __off_t;
__extension__ typedef __quad_t __off64_t;
__extension__ typedef int __pid_t;
__extension__ typedef struct { int __val[2]; } __fsid_t;
__extension__ typedef long int __clock_t;
__extension__ typedef unsigned long int __rlim_t;
__extension__ typedef __u_quad_t __rlim64_t;
__extension__ typedef unsigned int __id_t;
__extension__ typedef long int __time_t;
__extension__ typedef unsigned int __useconds_t;
__extension__ typedef long int __suseconds_t;

__extension__ typedef int __daddr_t;
__extension__ typedef long int __swblk_t;
__extension__ typedef int __key_t;


__extension__ typedef int __clockid_t;


__extension__ typedef void * __timer_t;


__extension__ typedef long int __blksize_t;




__extension__ typedef long int __blkcnt_t;
__extension__ typedef __quad_t __blkcnt64_t;


__extension__ typedef unsigned long int __fsblkcnt_t;
__extension__ typedef __u_quad_t __fsblkcnt64_t;


__extension__ typedef unsigned long int __fsfilcnt_t;
__extension__ typedef __u_quad_t __fsfilcnt64_t;

__extension__ typedef int __ssize_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


__extension__ typedef int __intptr_t;


__extension__ typedef unsigned int __socklen_t;
struct _IO_FILE;



typedef struct _IO_FILE FILE;
typedef struct _IO_FILE __FILE;




typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;

typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
typedef int _G_int16_t __attribute__ ((__mode__ (__HI__)));
typedef int _G_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int _G_uint16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int _G_uint32_t __attribute__ ((__mode__ (__SI__)));
typedef __builtin_va_list __gnuc_va_list;
struct _IO_jump_t; struct _IO_FILE;
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
struct _IO_FILE {
  int _flags;




  char* _IO_read_ptr;
  char* _IO_read_end;
  char* _IO_read_base;
  char* _IO_write_base;
  char* _IO_write_ptr;
  char* _IO_write_end;
  char* _IO_buf_base;
  char* _IO_buf_end;

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
 void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;
  size_t __pad5;

  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];

};


typedef struct _IO_FILE _IO_FILE;


struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, __const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);
extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __attribute__ ((__nothrow__));
extern int _IO_ferror (_IO_FILE *__fp) __attribute__ ((__nothrow__));

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern void _IO_funlockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern int _IO_ftrylockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __attribute__ ((__nothrow__));


typedef _G_fpos_t fpos_t;



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;
extern int remove (__const char *__filename) __attribute__ ((__nothrow__));

extern int rename (__const char *__old, __const char *__new) __attribute__ ((__nothrow__));
extern FILE *tmpfile (void) ;
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__)) ;





extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__)) ;
extern char *tempnam (__const char *__dir, __const char *__pfx)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;
extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);
extern int fflush_unlocked (FILE *__stream);






extern FILE *fopen (__const char *__restrict __filename,
      __const char *__restrict __modes) ;




extern FILE *freopen (__const char *__restrict __filename,
        __const char *__restrict __modes,
        FILE *__restrict __stream) ;
extern FILE *fdopen (int __fd, __const char *__modes) __attribute__ ((__nothrow__)) ;



extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__));



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__));





extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) __attribute__ ((__nothrow__));


extern void setlinebuf (FILE *__stream) __attribute__ ((__nothrow__));
extern int fprintf (FILE *__restrict __stream,
      __const char *__restrict __format, ...);




extern int printf (__const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      __const char *__restrict __format, ...) __attribute__ ((__nothrow__));





extern int vfprintf (FILE *__restrict __s, __const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (__const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, __const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));





extern int snprintf (char *__restrict __s, size_t __maxlen,
       __const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        __const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));





extern int fscanf (FILE *__restrict __stream,
     __const char *__restrict __format, ...) ;




extern int scanf (__const char *__restrict __format, ...) ;

extern int sscanf (__const char *__restrict __s,
     __const char *__restrict __format, ...) __attribute__ ((__nothrow__));





extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
extern int fgetc_unlocked (FILE *__stream);
extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);
extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;






extern char *gets (char *__s) ;





extern int fputs (__const char *__restrict __s, FILE *__restrict __stream);





extern int puts (__const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (__const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s) ;
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (__const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream) ;
extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, __const fpos_t *__pos);


extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__)) ;

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern void perror (__const char *__s);







extern int sys_nerr;
extern __const char *__const sys_errlist[];




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern FILE *popen (__const char *__command, __const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__));
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__)) ;


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__));
typedef long int wchar_t;


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
extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__)) ;




extern double atof (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





__extension__ extern long long int atoll (__const char *__nptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





extern double strtod (__const char *__restrict __nptr,
        char **__restrict __endptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;


extern long int strtol (__const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

extern unsigned long int strtoul (__const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




__extension__
extern long long int strtoq (__const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

__extension__
extern unsigned long long int strtouq (__const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





__extension__
extern long long int strtoll (__const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

__extension__
extern unsigned long long int strtoull (__const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern char *l64a (long int __n) __attribute__ ((__nothrow__)) ;


extern long int a64l (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;











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





typedef __off_t off_t;
typedef __pid_t pid_t;




typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;


typedef __time_t time_t;
typedef __clockid_t clockid_t;
typedef __timer_t timer_t;



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));




typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;



typedef __sigset_t sigset_t;






struct timespec
  {
    __time_t tv_sec;
    long int tv_nsec;
  };

struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };


typedef __suseconds_t suseconds_t;





typedef long int __fd_mask;
typedef struct
  {






    __fd_mask __fds_bits[1024 / (8 * sizeof (__fd_mask))];


  } fd_set;






typedef __fd_mask fd_mask;
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);


__extension__
extern unsigned int gnu_dev_major (unsigned long long int __dev)
     __attribute__ ((__nothrow__));
__extension__
extern unsigned int gnu_dev_minor (unsigned long long int __dev)
     __attribute__ ((__nothrow__));
__extension__
extern unsigned long long int gnu_dev_makedev (unsigned int __major,
            unsigned int __minor)
     __attribute__ ((__nothrow__));
typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
typedef unsigned long int pthread_t;


typedef union
{
  char __size[36];
  long int __align;
} pthread_attr_t;


typedef struct __pthread_internal_slist
{
  struct __pthread_internal_slist *__next;
} __pthread_slist_t;




typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;


    int __kind;
    unsigned int __nusers;
    __extension__ union
    {
      int __spins;
      __pthread_slist_t __list;
    };
  } __data;
  char __size[24];
  long int __align;
} pthread_mutex_t;

typedef union
{
  char __size[4];
  long int __align;
} pthread_mutexattr_t;




typedef union
{
  struct
  {
    int __lock;
    unsigned int __futex;
    __extension__ unsigned long long int __total_seq;
    __extension__ unsigned long long int __wakeup_seq;
    __extension__ unsigned long long int __woken_seq;
    void *__mutex;
    unsigned int __nwaiters;
    unsigned int __broadcast_seq;
  } __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;

typedef union
{
  char __size[4];
  long int __align;
} pthread_condattr_t;



typedef unsigned int pthread_key_t;



typedef int pthread_once_t;





typedef union
{
  struct
  {
    int __lock;
    unsigned int __nr_readers;
    unsigned int __readers_wakeup;
    unsigned int __writer_wakeup;
    unsigned int __nr_readers_queued;
    unsigned int __nr_writers_queued;


    unsigned char __flags;
    unsigned char __shared;
    unsigned char __pad1;
    unsigned char __pad2;
    int __writer;
  } __data;
  char __size[32];
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
  char __size[20];
  long int __align;
} pthread_barrier_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;






extern long int random (void) __attribute__ ((__nothrow__));


extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__));





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







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

extern int random_r (struct random_data *__restrict __buf,
       int32_t *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));






extern int rand (void) __attribute__ ((__nothrow__));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__));




extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__));







extern double drand48 (void) __attribute__ ((__nothrow__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) __attribute__ ((__nothrow__));
extern long int nrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) __attribute__ ((__nothrow__));
extern long int jrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) __attribute__ ((__nothrow__));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    unsigned long long int __a;
  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern void *malloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;

extern void *calloc (size_t __nmemb, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;
extern void *realloc (void *__ptr, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) __attribute__ ((__nothrow__));




extern void cfree (void *__ptr) __attribute__ ((__nothrow__));











extern void *alloca (size_t __size) __attribute__ ((__nothrow__));




extern void *valloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern void abort (void) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));


extern char *getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern char *__secure_getenv (__const char *__name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int putenv (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (__const char *__name, __const char *__value, int __replace)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (__const char *__name) __attribute__ ((__nothrow__));






extern int clearenv (void) __attribute__ ((__nothrow__));
extern char *mktemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int system (__const char *__command) ;
extern char *realpath (__const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__)) ;






typedef int (*__compar_fn_t) (__const void *, __const void *);



extern void *bsearch (__const void *__key, __const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;



extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
extern int abs (int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));







extern int mblen (__const char *__s, size_t __n) __attribute__ ((__nothrow__)) ;


extern int mbtowc (wchar_t *__restrict __pwc,
     __const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__)) ;


extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__)) ;



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   __const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__));

extern size_t wcstombs (char *__restrict __s,
   __const wchar_t *__restrict __pwcs, size_t __n)
     __attribute__ ((__nothrow__));
extern int rpmatch (__const char *__response) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int posix_openpt (int __oflag) ;
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;

typedef unsigned int uint32_t;





__extension__
typedef unsigned long long int uint64_t;






typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;



__extension__
typedef long long int int_least64_t;



typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;



__extension__
typedef unsigned long long int uint_least64_t;






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
__extension__
typedef long long int intmax_t;
__extension__
typedef unsigned long long int uintmax_t;









extern void *memcpy (void *__restrict __dest,
       __const void *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern void *memmove (void *__dest, __const void *__src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));






extern void *memccpy (void *__restrict __dest, __const void *__restrict __src,
        int __c, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));





extern void *memset (void *__s, int __c, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int memcmp (__const void *__s1, __const void *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern void *memchr (__const void *__s, int __c, size_t __n)
      __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


extern char *strcpy (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strncpy (char *__restrict __dest,
        __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *strcat (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strncat (char *__restrict __dest, __const char *__restrict __src,
        size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern int strncmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strcoll (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern size_t strxfrm (char *__restrict __dest,
         __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
extern char *strdup (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));


extern char *strchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern char *strrchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));



extern size_t strcspn (__const char *__s, __const char *__reject)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern size_t strspn (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strpbrk (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));

extern char *strstr (__const char *__haystack, __const char *__needle)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));



extern char *strtok (char *__restrict __s, __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));




extern char *__strtok_r (char *__restrict __s,
    __const char *__restrict __delim,
    char **__restrict __save_ptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));

extern char *strtok_r (char *__restrict __s, __const char *__restrict __delim,
         char **__restrict __save_ptr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));


extern size_t strlen (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


extern char *strerror (int __errnum) __attribute__ ((__nothrow__));
extern int strerror_r (int __errnum, char *__buf, size_t __buflen) __asm__ ("" "__xpg_strerror_r") __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
extern void __bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern void bcopy (__const void *__src, void *__dest, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern void bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int bcmp (__const void *__s1, __const void *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *index (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


extern char *rindex (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));



extern int ffs (int __i) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
extern int strcasecmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
extern char *strsep (char **__restrict __stringp,
       __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
typedef __useconds_t useconds_t;
typedef __socklen_t socklen_t;
extern int access (__const char *__name, int __type) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern __off_t lseek (int __fd, __off_t __offset, int __whence) __attribute__ ((__nothrow__));
extern int close (int __fd);






extern ssize_t read (int __fd, void *__buf, size_t __nbytes) ;





extern ssize_t write (int __fd, __const void *__buf, size_t __n) ;
extern int pipe (int __pipedes[2]) __attribute__ ((__nothrow__)) ;
extern unsigned int alarm (unsigned int __seconds) __attribute__ ((__nothrow__));
extern unsigned int sleep (unsigned int __seconds);






extern __useconds_t ualarm (__useconds_t __value, __useconds_t __interval)
     __attribute__ ((__nothrow__));






extern int usleep (__useconds_t __useconds);
extern int pause (void);



extern int chown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchown (int __fd, __uid_t __owner, __gid_t __group) __attribute__ ((__nothrow__)) ;




extern int lchown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int chdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchdir (int __fd) __attribute__ ((__nothrow__)) ;
extern char *getcwd (char *__buf, size_t __size) __attribute__ ((__nothrow__)) ;
extern char *getwd (char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__)) ;




extern int dup (int __fd) __attribute__ ((__nothrow__)) ;


extern int dup2 (int __fd, int __fd2) __attribute__ ((__nothrow__));
extern char **__environ;







extern int execve (__const char *__path, char *__const __argv[],
     char *__const __envp[]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int execv (__const char *__path, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execle (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execl (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int execvp (__const char *__file, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int execlp (__const char *__file, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int nice (int __inc) __attribute__ ((__nothrow__)) ;




extern void _exit (int __status) __attribute__ ((__noreturn__));






enum
  {
    _PC_LINK_MAX,

    _PC_MAX_CANON,

    _PC_MAX_INPUT,

    _PC_NAME_MAX,

    _PC_PATH_MAX,

    _PC_PIPE_BUF,

    _PC_CHOWN_RESTRICTED,

    _PC_NO_TRUNC,

    _PC_VDISABLE,

    _PC_SYNC_IO,

    _PC_ASYNC_IO,

    _PC_PRIO_IO,

    _PC_SOCK_MAXBUF,

    _PC_FILESIZEBITS,

    _PC_REC_INCR_XFER_SIZE,

    _PC_REC_MAX_XFER_SIZE,

    _PC_REC_MIN_XFER_SIZE,

    _PC_REC_XFER_ALIGN,

    _PC_ALLOC_SIZE_MIN,

    _PC_SYMLINK_MAX,

    _PC_2_SYMLINKS

  };


enum
  {
    _SC_ARG_MAX,

    _SC_CHILD_MAX,

    _SC_CLK_TCK,

    _SC_NGROUPS_MAX,

    _SC_OPEN_MAX,

    _SC_STREAM_MAX,

    _SC_TZNAME_MAX,

    _SC_JOB_CONTROL,

    _SC_SAVED_IDS,

    _SC_REALTIME_SIGNALS,

    _SC_PRIORITY_SCHEDULING,

    _SC_TIMERS,

    _SC_ASYNCHRONOUS_IO,

    _SC_PRIORITIZED_IO,

    _SC_SYNCHRONIZED_IO,

    _SC_FSYNC,

    _SC_MAPPED_FILES,

    _SC_MEMLOCK,

    _SC_MEMLOCK_RANGE,

    _SC_MEMORY_PROTECTION,

    _SC_MESSAGE_PASSING,

    _SC_SEMAPHORES,

    _SC_SHARED_MEMORY_OBJECTS,

    _SC_AIO_LISTIO_MAX,

    _SC_AIO_MAX,

    _SC_AIO_PRIO_DELTA_MAX,

    _SC_DELAYTIMER_MAX,

    _SC_MQ_OPEN_MAX,

    _SC_MQ_PRIO_MAX,

    _SC_VERSION,

    _SC_PAGESIZE,


    _SC_RTSIG_MAX,

    _SC_SEM_NSEMS_MAX,

    _SC_SEM_VALUE_MAX,

    _SC_SIGQUEUE_MAX,

    _SC_TIMER_MAX,




    _SC_BC_BASE_MAX,

    _SC_BC_DIM_MAX,

    _SC_BC_SCALE_MAX,

    _SC_BC_STRING_MAX,

    _SC_COLL_WEIGHTS_MAX,

    _SC_EQUIV_CLASS_MAX,

    _SC_EXPR_NEST_MAX,

    _SC_LINE_MAX,

    _SC_RE_DUP_MAX,

    _SC_CHARCLASS_NAME_MAX,


    _SC_2_VERSION,

    _SC_2_C_BIND,

    _SC_2_C_DEV,

    _SC_2_FORT_DEV,

    _SC_2_FORT_RUN,

    _SC_2_SW_DEV,

    _SC_2_LOCALEDEF,


    _SC_PII,

    _SC_PII_XTI,

    _SC_PII_SOCKET,

    _SC_PII_INTERNET,

    _SC_PII_OSI,

    _SC_POLL,

    _SC_SELECT,

    _SC_UIO_MAXIOV,

    _SC_IOV_MAX = _SC_UIO_MAXIOV,

    _SC_PII_INTERNET_STREAM,

    _SC_PII_INTERNET_DGRAM,

    _SC_PII_OSI_COTS,

    _SC_PII_OSI_CLTS,

    _SC_PII_OSI_M,

    _SC_T_IOV_MAX,



    _SC_THREADS,

    _SC_THREAD_SAFE_FUNCTIONS,

    _SC_GETGR_R_SIZE_MAX,

    _SC_GETPW_R_SIZE_MAX,

    _SC_LOGIN_NAME_MAX,

    _SC_TTY_NAME_MAX,

    _SC_THREAD_DESTRUCTOR_ITERATIONS,

    _SC_THREAD_KEYS_MAX,

    _SC_THREAD_STACK_MIN,

    _SC_THREAD_THREADS_MAX,

    _SC_THREAD_ATTR_STACKADDR,

    _SC_THREAD_ATTR_STACKSIZE,

    _SC_THREAD_PRIORITY_SCHEDULING,

    _SC_THREAD_PRIO_INHERIT,

    _SC_THREAD_PRIO_PROTECT,

    _SC_THREAD_PROCESS_SHARED,


    _SC_NPROCESSORS_CONF,

    _SC_NPROCESSORS_ONLN,

    _SC_PHYS_PAGES,

    _SC_AVPHYS_PAGES,

    _SC_ATEXIT_MAX,

    _SC_PASS_MAX,


    _SC_XOPEN_VERSION,

    _SC_XOPEN_XCU_VERSION,

    _SC_XOPEN_UNIX,

    _SC_XOPEN_CRYPT,

    _SC_XOPEN_ENH_I18N,

    _SC_XOPEN_SHM,


    _SC_2_CHAR_TERM,

    _SC_2_C_VERSION,

    _SC_2_UPE,


    _SC_XOPEN_XPG2,

    _SC_XOPEN_XPG3,

    _SC_XOPEN_XPG4,


    _SC_CHAR_BIT,

    _SC_CHAR_MAX,

    _SC_CHAR_MIN,

    _SC_INT_MAX,

    _SC_INT_MIN,

    _SC_LONG_BIT,

    _SC_WORD_BIT,

    _SC_MB_LEN_MAX,

    _SC_NZERO,

    _SC_SSIZE_MAX,

    _SC_SCHAR_MAX,

    _SC_SCHAR_MIN,

    _SC_SHRT_MAX,

    _SC_SHRT_MIN,

    _SC_UCHAR_MAX,

    _SC_UINT_MAX,

    _SC_ULONG_MAX,

    _SC_USHRT_MAX,


    _SC_NL_ARGMAX,

    _SC_NL_LANGMAX,

    _SC_NL_MSGMAX,

    _SC_NL_NMAX,

    _SC_NL_SETMAX,

    _SC_NL_TEXTMAX,


    _SC_XBS5_ILP32_OFF32,

    _SC_XBS5_ILP32_OFFBIG,

    _SC_XBS5_LP64_OFF64,

    _SC_XBS5_LPBIG_OFFBIG,


    _SC_XOPEN_LEGACY,

    _SC_XOPEN_REALTIME,

    _SC_XOPEN_REALTIME_THREADS,


    _SC_ADVISORY_INFO,

    _SC_BARRIERS,

    _SC_BASE,

    _SC_C_LANG_SUPPORT,

    _SC_C_LANG_SUPPORT_R,

    _SC_CLOCK_SELECTION,

    _SC_CPUTIME,

    _SC_THREAD_CPUTIME,

    _SC_DEVICE_IO,

    _SC_DEVICE_SPECIFIC,

    _SC_DEVICE_SPECIFIC_R,

    _SC_FD_MGMT,

    _SC_FIFO,

    _SC_PIPE,

    _SC_FILE_ATTRIBUTES,

    _SC_FILE_LOCKING,

    _SC_FILE_SYSTEM,

    _SC_MONOTONIC_CLOCK,

    _SC_MULTI_PROCESS,

    _SC_SINGLE_PROCESS,

    _SC_NETWORKING,

    _SC_READER_WRITER_LOCKS,

    _SC_SPIN_LOCKS,

    _SC_REGEXP,

    _SC_REGEX_VERSION,

    _SC_SHELL,

    _SC_SIGNALS,

    _SC_SPAWN,

    _SC_SPORADIC_SERVER,

    _SC_THREAD_SPORADIC_SERVER,

    _SC_SYSTEM_DATABASE,

    _SC_SYSTEM_DATABASE_R,

    _SC_TIMEOUTS,

    _SC_TYPED_MEMORY_OBJECTS,

    _SC_USER_GROUPS,

    _SC_USER_GROUPS_R,

    _SC_2_PBS,

    _SC_2_PBS_ACCOUNTING,

    _SC_2_PBS_LOCATE,

    _SC_2_PBS_MESSAGE,

    _SC_2_PBS_TRACK,

    _SC_SYMLOOP_MAX,

    _SC_STREAMS,

    _SC_2_PBS_CHECKPOINT,


    _SC_V6_ILP32_OFF32,

    _SC_V6_ILP32_OFFBIG,

    _SC_V6_LP64_OFF64,

    _SC_V6_LPBIG_OFFBIG,


    _SC_HOST_NAME_MAX,

    _SC_TRACE,

    _SC_TRACE_EVENT_FILTER,

    _SC_TRACE_INHERIT,

    _SC_TRACE_LOG,


    _SC_LEVEL1_ICACHE_SIZE,

    _SC_LEVEL1_ICACHE_ASSOC,

    _SC_LEVEL1_ICACHE_LINESIZE,

    _SC_LEVEL1_DCACHE_SIZE,

    _SC_LEVEL1_DCACHE_ASSOC,

    _SC_LEVEL1_DCACHE_LINESIZE,

    _SC_LEVEL2_CACHE_SIZE,

    _SC_LEVEL2_CACHE_ASSOC,

    _SC_LEVEL2_CACHE_LINESIZE,

    _SC_LEVEL3_CACHE_SIZE,

    _SC_LEVEL3_CACHE_ASSOC,

    _SC_LEVEL3_CACHE_LINESIZE,

    _SC_LEVEL4_CACHE_SIZE,

    _SC_LEVEL4_CACHE_ASSOC,

    _SC_LEVEL4_CACHE_LINESIZE,



    _SC_IPV6 = _SC_LEVEL1_ICACHE_SIZE + 50,

    _SC_RAW_SOCKETS

  };


enum
  {
    _CS_PATH,


    _CS_V6_WIDTH_RESTRICTED_ENVS,



    _CS_GNU_LIBC_VERSION,

    _CS_GNU_LIBPTHREAD_VERSION,


    _CS_LFS_CFLAGS = 1000,

    _CS_LFS_LDFLAGS,

    _CS_LFS_LIBS,

    _CS_LFS_LINTFLAGS,

    _CS_LFS64_CFLAGS,

    _CS_LFS64_LDFLAGS,

    _CS_LFS64_LIBS,

    _CS_LFS64_LINTFLAGS,


    _CS_XBS5_ILP32_OFF32_CFLAGS = 1100,

    _CS_XBS5_ILP32_OFF32_LDFLAGS,

    _CS_XBS5_ILP32_OFF32_LIBS,

    _CS_XBS5_ILP32_OFF32_LINTFLAGS,

    _CS_XBS5_ILP32_OFFBIG_CFLAGS,

    _CS_XBS5_ILP32_OFFBIG_LDFLAGS,

    _CS_XBS5_ILP32_OFFBIG_LIBS,

    _CS_XBS5_ILP32_OFFBIG_LINTFLAGS,

    _CS_XBS5_LP64_OFF64_CFLAGS,

    _CS_XBS5_LP64_OFF64_LDFLAGS,

    _CS_XBS5_LP64_OFF64_LIBS,

    _CS_XBS5_LP64_OFF64_LINTFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_CFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_LDFLAGS,

    _CS_XBS5_LPBIG_OFFBIG_LIBS,

    _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,


    _CS_POSIX_V6_ILP32_OFF32_CFLAGS,

    _CS_POSIX_V6_ILP32_OFF32_LDFLAGS,

    _CS_POSIX_V6_ILP32_OFF32_LIBS,

    _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,

    _CS_POSIX_V6_ILP32_OFFBIG_LIBS,

    _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,

    _CS_POSIX_V6_LP64_OFF64_CFLAGS,

    _CS_POSIX_V6_LP64_OFF64_LDFLAGS,

    _CS_POSIX_V6_LP64_OFF64_LIBS,

    _CS_POSIX_V6_LP64_OFF64_LINTFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LIBS,

    _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS

  };


extern long int pathconf (__const char *__path, int __name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int fpathconf (int __fd, int __name) __attribute__ ((__nothrow__));


extern long int sysconf (int __name) __attribute__ ((__nothrow__));



extern size_t confstr (int __name, char *__buf, size_t __len) __attribute__ ((__nothrow__));




extern __pid_t getpid (void) __attribute__ ((__nothrow__));


extern __pid_t getppid (void) __attribute__ ((__nothrow__));




extern __pid_t getpgrp (void) __attribute__ ((__nothrow__));
extern __pid_t __getpgid (__pid_t __pid) __attribute__ ((__nothrow__));
extern int setpgid (__pid_t __pid, __pid_t __pgid) __attribute__ ((__nothrow__));
extern int setpgrp (void) __attribute__ ((__nothrow__));
extern __pid_t setsid (void) __attribute__ ((__nothrow__));







extern __uid_t getuid (void) __attribute__ ((__nothrow__));


extern __uid_t geteuid (void) __attribute__ ((__nothrow__));


extern __gid_t getgid (void) __attribute__ ((__nothrow__));


extern __gid_t getegid (void) __attribute__ ((__nothrow__));




extern int getgroups (int __size, __gid_t __list[]) __attribute__ ((__nothrow__)) ;
extern int setuid (__uid_t __uid) __attribute__ ((__nothrow__));




extern int setreuid (__uid_t __ruid, __uid_t __euid) __attribute__ ((__nothrow__));




extern int seteuid (__uid_t __uid) __attribute__ ((__nothrow__));






extern int setgid (__gid_t __gid) __attribute__ ((__nothrow__));




extern int setregid (__gid_t __rgid, __gid_t __egid) __attribute__ ((__nothrow__));




extern int setegid (__gid_t __gid) __attribute__ ((__nothrow__));
extern __pid_t fork (void) __attribute__ ((__nothrow__));






extern __pid_t vfork (void) __attribute__ ((__nothrow__));





extern char *ttyname (int __fd) __attribute__ ((__nothrow__));



extern int ttyname_r (int __fd, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;



extern int isatty (int __fd) __attribute__ ((__nothrow__));





extern int ttyslot (void) __attribute__ ((__nothrow__));




extern int link (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
extern int symlink (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern ssize_t readlink (__const char *__restrict __path,
    char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
extern int unlink (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int rmdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern __pid_t tcgetpgrp (int __fd) __attribute__ ((__nothrow__));


extern int tcsetpgrp (int __fd, __pid_t __pgrp_id) __attribute__ ((__nothrow__));






extern char *getlogin (void);







extern int getlogin_r (char *__name, size_t __name_len) __attribute__ ((__nonnull__ (1)));




extern int setlogin (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern char *optarg;
extern int optind;




extern int opterr;



extern int optopt;
extern int getopt (int ___argc, char *const *___argv, const char *__shortopts)
       __attribute__ ((__nothrow__));







extern int gethostname (char *__name, size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int sethostname (__const char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int sethostid (long int __id) __attribute__ ((__nothrow__)) ;





extern int getdomainname (char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int setdomainname (__const char *__name, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int vhangup (void) __attribute__ ((__nothrow__));


extern int revoke (__const char *__file) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;







extern int profil (unsigned short int *__sample_buffer, size_t __size,
     size_t __offset, unsigned int __scale)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int acct (__const char *__name) __attribute__ ((__nothrow__));



extern char *getusershell (void) __attribute__ ((__nothrow__));
extern void endusershell (void) __attribute__ ((__nothrow__));
extern void setusershell (void) __attribute__ ((__nothrow__));





extern int daemon (int __nochdir, int __noclose) __attribute__ ((__nothrow__)) ;






extern int chroot (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern char *getpass (__const char *__prompt) __attribute__ ((__nonnull__ (1)));
extern int fsync (int __fd);






extern long int gethostid (void);


extern void sync (void) __attribute__ ((__nothrow__));




extern int getpagesize (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int getdtablesize (void) __attribute__ ((__nothrow__));




extern int truncate (__const char *__file, __off_t __length)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
extern int ftruncate (int __fd, __off_t __length) __attribute__ ((__nothrow__)) ;
extern int brk (void *__addr) __attribute__ ((__nothrow__)) ;





extern void *sbrk (intptr_t __delta) __attribute__ ((__nothrow__));
extern long int syscall (long int __sysno, ...) __attribute__ ((__nothrow__));
extern int lockf (int __fd, int __cmd, __off_t __len) ;
extern int fdatasync (int __fildes);













extern int *__errno_location (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int __sigismember (__const __sigset_t *, int);
extern int __sigaddset (__sigset_t *, int);
extern int __sigdelset (__sigset_t *, int);







typedef __sig_atomic_t sig_atomic_t;
typedef void (*__sighandler_t) (int);




extern __sighandler_t __sysv_signal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));


extern __sighandler_t signal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));
extern int kill (__pid_t __pid, int __sig) __attribute__ ((__nothrow__));






extern int killpg (__pid_t __pgrp, int __sig) __attribute__ ((__nothrow__));




extern int raise (int __sig) __attribute__ ((__nothrow__));




extern __sighandler_t ssignal (int __sig, __sighandler_t __handler)
     __attribute__ ((__nothrow__));
extern int gsignal (int __sig) __attribute__ ((__nothrow__));




extern void psignal (int __sig, __const char *__s);
extern int __sigpause (int __sig_or_mask, int __is_sig);
extern int sigblock (int __mask) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));


extern int sigsetmask (int __mask) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));


extern int siggetmask (void) __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));
typedef __sighandler_t sig_t;

















typedef union sigval
  {
    int sival_int;
    void *sival_ptr;
  } sigval_t;
typedef struct siginfo
  {
    int si_signo;
    int si_errno;

    int si_code;

    union
      {
 int _pad[((128 / sizeof (int)) - 3)];


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
   } _kill;


 struct
   {
     int si_tid;
     int si_overrun;
     sigval_t si_sigval;
   } _timer;


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
     sigval_t si_sigval;
   } _rt;


 struct
   {
     __pid_t si_pid;
     __uid_t si_uid;
     int si_status;
     __clock_t si_utime;
     __clock_t si_stime;
   } _sigchld;


 struct
   {
     void *si_addr;
   } _sigfault;


 struct
   {
     long int si_band;
     int si_fd;
   } _sigpoll;
      } _sifields;
  } siginfo_t;
enum
{
  SI_ASYNCNL = -60,

  SI_TKILL = -6,

  SI_SIGIO,

  SI_ASYNCIO,

  SI_MESGQ,

  SI_TIMER,

  SI_QUEUE,

  SI_USER,

  SI_KERNEL = 0x80

};



enum
{
  ILL_ILLOPC = 1,

  ILL_ILLOPN,

  ILL_ILLADR,

  ILL_ILLTRP,

  ILL_PRVOPC,

  ILL_PRVREG,

  ILL_COPROC,

  ILL_BADSTK

};


enum
{
  FPE_INTDIV = 1,

  FPE_INTOVF,

  FPE_FLTDIV,

  FPE_FLTOVF,

  FPE_FLTUND,

  FPE_FLTRES,

  FPE_FLTINV,

  FPE_FLTSUB

};


enum
{
  SEGV_MAPERR = 1,

  SEGV_ACCERR

};


enum
{
  BUS_ADRALN = 1,

  BUS_ADRERR,

  BUS_OBJERR

};


enum
{
  TRAP_BRKPT = 1,

  TRAP_TRACE

};


enum
{
  CLD_EXITED = 1,

  CLD_KILLED,

  CLD_DUMPED,

  CLD_TRAPPED,

  CLD_STOPPED,

  CLD_CONTINUED

};


enum
{
  POLL_IN = 1,

  POLL_OUT,

  POLL_MSG,

  POLL_ERR,

  POLL_PRI,

  POLL_HUP

};
typedef struct sigevent
  {
    sigval_t sigev_value;
    int sigev_signo;
    int sigev_notify;

    union
      {
 int _pad[((64 / sizeof (int)) - 3)];



 __pid_t _tid;

 struct
   {
     void (*_function) (sigval_t);
     void *_attribute;
   } _sigev_thread;
      } _sigev_un;
  } sigevent_t;






enum
{
  SIGEV_SIGNAL = 0,

  SIGEV_NONE,

  SIGEV_THREAD,


  SIGEV_THREAD_ID = 4

};



extern int sigemptyset (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigfillset (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigaddset (sigset_t *__set, int __signo) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigdelset (sigset_t *__set, int __signo) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int sigismember (__const sigset_t *__set, int __signo)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
struct sigaction
  {


    union
      {

 __sighandler_t sa_handler;

 void (*sa_sigaction) (int, siginfo_t *, void *);
      }
    __sigaction_handler;







    __sigset_t sa_mask;


    int sa_flags;


    void (*sa_restorer) (void);
  };


extern int sigprocmask (int __how, __const sigset_t *__restrict __set,
   sigset_t *__restrict __oset) __attribute__ ((__nothrow__));






extern int sigsuspend (__const sigset_t *__set) __attribute__ ((__nonnull__ (1)));


extern int sigaction (int __sig, __const struct sigaction *__restrict __act,
        struct sigaction *__restrict __oact) __attribute__ ((__nothrow__));


extern int sigpending (sigset_t *__set) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int sigwait (__const sigset_t *__restrict __set, int *__restrict __sig)
     __attribute__ ((__nonnull__ (1, 2)));






extern int sigwaitinfo (__const sigset_t *__restrict __set,
   siginfo_t *__restrict __info) __attribute__ ((__nonnull__ (1)));






extern int sigtimedwait (__const sigset_t *__restrict __set,
    siginfo_t *__restrict __info,
    __const struct timespec *__restrict __timeout)
     __attribute__ ((__nonnull__ (1)));



extern int sigqueue (__pid_t __pid, int __sig, __const union sigval __val)
     __attribute__ ((__nothrow__));
extern __const char *__const _sys_siglist[65];
extern __const char *__const sys_siglist[65];


struct sigvec
  {
    __sighandler_t sv_handler;
    int sv_mask;

    int sv_flags;

  };
extern int sigvec (int __sig, __const struct sigvec *__vec,
     struct sigvec *__ovec) __attribute__ ((__nothrow__));




struct _fpreg {
 unsigned short significand[4];
 unsigned short exponent;
};

struct _fpxreg {
 unsigned short significand[4];
 unsigned short exponent;
 unsigned short padding[3];
};

struct _xmmreg {
 unsigned long element[4];
};

struct _fpstate {

 unsigned long cw;
 unsigned long sw;
 unsigned long tag;
 unsigned long ipoff;
 unsigned long cssel;
 unsigned long dataoff;
 unsigned long datasel;
 struct _fpreg _st[8];
 unsigned short status;
 unsigned short magic;


 unsigned long _fxsr_env[6];
 unsigned long mxcsr;
 unsigned long reserved;
 struct _fpxreg _fxsr_st[8];
 struct _xmmreg _xmm[8];
 unsigned long padding[56];
};



struct sigcontext {
 unsigned short gs, __gsh;
 unsigned short fs, __fsh;
 unsigned short es, __esh;
 unsigned short ds, __dsh;
 unsigned long edi;
 unsigned long esi;
 unsigned long ebp;
 unsigned long esp;
 unsigned long ebx;
 unsigned long edx;
 unsigned long ecx;
 unsigned long eax;
 unsigned long trapno;
 unsigned long err;
 unsigned long eip;
 unsigned short cs, __csh;
 unsigned long eflags;
 unsigned long esp_at_signal;
 unsigned short ss, __ssh;
 struct _fpstate * fpstate;
 unsigned long oldmask;
 unsigned long cr2;
};


extern int sigreturn (struct sigcontext *__scp) __attribute__ ((__nothrow__));











extern int siginterrupt (int __sig, int __interrupt) __attribute__ ((__nothrow__));


struct sigstack
  {
    void *ss_sp;
    int ss_onstack;
  };



enum
{
  SS_ONSTACK = 1,

  SS_DISABLE

};
typedef struct sigaltstack
  {
    void *ss_sp;
    int ss_flags;
    size_t ss_size;
  } stack_t;
extern int sigstack (struct sigstack *__ss, struct sigstack *__oss)
     __attribute__ ((__nothrow__)) __attribute__ ((__deprecated__));



extern int sigaltstack (__const struct sigaltstack *__restrict __ss,
   struct sigaltstack *__restrict __oss) __attribute__ ((__nothrow__));
extern int pthread_sigmask (int __how,
       __const __sigset_t *__restrict __newmask,
       __sigset_t *__restrict __oldmask)__attribute__ ((__nothrow__));


extern int pthread_kill (pthread_t __threadid, int __signo) __attribute__ ((__nothrow__));






extern int __libc_current_sigrtmin (void) __attribute__ ((__nothrow__));

extern int __libc_current_sigrtmax (void) __attribute__ ((__nothrow__));




struct stat
  {
    __dev_t st_dev;
    unsigned short int __pad1;

    __ino_t st_ino;



    __mode_t st_mode;
    __nlink_t st_nlink;
    __uid_t st_uid;
    __gid_t st_gid;
    __dev_t st_rdev;
    unsigned short int __pad2;

    __off_t st_size;



    __blksize_t st_blksize;


    __blkcnt_t st_blocks;
 struct timespec st_atim;
    struct timespec st_mtim;
    struct timespec st_ctim;
 unsigned long int __unused4;
    unsigned long int __unused5;



  };
extern int stat (__const char *__restrict __file,
   struct stat *__restrict __buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int fstat (int __fd, struct stat *__buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
extern int lstat (__const char *__restrict __file,
    struct stat *__restrict __buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int chmod (__const char *__file, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int lchmod (__const char *__file, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int fchmod (int __fd, __mode_t __mode) __attribute__ ((__nothrow__));
extern __mode_t umask (__mode_t __mask) __attribute__ ((__nothrow__));
extern int mkdir (__const char *__path, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int mknod (__const char *__path, __mode_t __mode, __dev_t __dev)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int mkfifo (__const char *__path, __mode_t __mode)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int __fxstat (int __ver, int __fildes, struct stat *__stat_buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));
extern int __xstat (int __ver, __const char *__filename,
      struct stat *__stat_buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));
extern int __lxstat (int __ver, __const char *__filename,
       struct stat *__stat_buf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));
extern int __fxstatat (int __ver, int __fildes, __const char *__filename,
         struct stat *__stat_buf, int __flag)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));
extern int __xmknod (int __ver, __const char *__path, __mode_t __mode,
       __dev_t *__dev) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));

extern int __xmknodat (int __ver, int __fd, __const char *__path,
         __mode_t __mode, __dev_t *__dev)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 5)));








struct winsize
  {
    unsigned short int ws_row;
    unsigned short int ws_col;
    unsigned short int ws_xpixel;
    unsigned short int ws_ypixel;
  };


struct termio
  {
    unsigned short int c_iflag;
    unsigned short int c_oflag;
    unsigned short int c_cflag;
    unsigned short int c_lflag;
    unsigned char c_line;
    unsigned char c_cc[8];
};










extern int ioctl (int __fd, unsigned long int __request, ...) __attribute__ ((__nothrow__));





struct flock
  {
    short int l_type;
    short int l_whence;

    __off_t l_start;
    __off_t l_len;




    __pid_t l_pid;
  };
extern int fcntl (int __fd, int __cmd, ...);
extern int open (__const char *__file, int __oflag, ...) __attribute__ ((__nonnull__ (1)));
extern int creat (__const char *__file, __mode_t __mode) __attribute__ ((__nonnull__ (1)));
extern int posix_fadvise (int __fd, __off_t __offset, __off_t __len,
     int __advise) __attribute__ ((__nothrow__));
extern int posix_fallocate (int __fd, __off_t __offset, __off_t __len);









typedef unsigned short umode_t;






typedef __signed__ char __s8;
typedef unsigned char __u8;

typedef __signed__ short __s16;
typedef unsigned short __u16;

typedef __signed__ int __s32;
typedef unsigned int __u32;


typedef __signed__ long long __s64;
typedef unsigned long long __u64;
struct fb_fix_screeninfo {
 char id[16];
 unsigned long smem_start;

 __u32 smem_len;
 __u32 type;
 __u32 type_aux;
 __u32 visual;
 __u16 xpanstep;
 __u16 ypanstep;
 __u16 ywrapstep;
 __u32 line_length;
 unsigned long mmio_start;

 __u32 mmio_len;
 __u32 accel;

 __u16 reserved[3];
};







struct fb_bitfield {
 __u32 offset;
 __u32 length;
 __u32 msb_right;

};
struct fb_var_screeninfo {
 __u32 xres;
 __u32 yres;
 __u32 xres_virtual;
 __u32 yres_virtual;
 __u32 xoffset;
 __u32 yoffset;

 __u32 bits_per_pixel;
 __u32 grayscale;

 struct fb_bitfield red;
 struct fb_bitfield green;
 struct fb_bitfield blue;
 struct fb_bitfield transp;

 __u32 nonstd;

 __u32 activate;

 __u32 height;
 __u32 width;

 __u32 accel_flags;


 __u32 pixclock;
 __u32 left_margin;
 __u32 right_margin;
 __u32 upper_margin;
 __u32 lower_margin;
 __u32 hsync_len;
 __u32 vsync_len;
 __u32 sync;
 __u32 vmode;
 __u32 rotate;
 __u32 reserved[5];
};

struct fb_cmap {
 __u32 start;
 __u32 len;
 __u16 *red;
 __u16 *green;
 __u16 *blue;
 __u16 *transp;
};

struct fb_con2fbmap {
 __u32 console;
 __u32 framebuffer;
};
enum {

 FB_BLANK_UNBLANK = 0,


 FB_BLANK_NORMAL = 0 + 1,


 FB_BLANK_VSYNC_SUSPEND = 1 + 1,


 FB_BLANK_HSYNC_SUSPEND = 2 + 1,


 FB_BLANK_POWERDOWN = 3 + 1
};
struct fb_vblank {
 __u32 flags;
 __u32 count;
 __u32 vcount;
 __u32 hcount;
 __u32 reserved[4];
};





struct fb_copyarea {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 sx;
 __u32 sy;
};

struct fb_fillrect {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 color;
 __u32 rop;
};

struct fb_image {
 __u32 dx;
 __u32 dy;
 __u32 width;
 __u32 height;
 __u32 fg_color;
 __u32 bg_color;
 __u8 depth;
 const char *data;
 struct fb_cmap cmap;
};
struct fbcurpos {
 __u16 x, y;
};

struct fb_cursor {
 __u16 set;
 __u16 enable;
 __u16 rop;
 const char *mask;
 struct fbcurpos hot;
 struct fb_image image;
};
extern void *mmap (void *__addr, size_t __len, int __prot,
     int __flags, int __fd, __off_t __offset) __attribute__ ((__nothrow__));
extern int munmap (void *__addr, size_t __len) __attribute__ ((__nothrow__));




extern int mprotect (void *__addr, size_t __len, int __prot) __attribute__ ((__nothrow__));







extern int msync (void *__addr, size_t __len, int __flags);




extern int madvise (void *__addr, size_t __len, int __advice) __attribute__ ((__nothrow__));



extern int posix_madvise (void *__addr, size_t __len, int __advice) __attribute__ ((__nothrow__));




extern int mlock (__const void *__addr, size_t __len) __attribute__ ((__nothrow__));


extern int munlock (__const void *__addr, size_t __len) __attribute__ ((__nothrow__));




extern int mlockall (int __flags) __attribute__ ((__nothrow__));



extern int munlockall (void) __attribute__ ((__nothrow__));







extern int mincore (void *__start, size_t __len, unsigned char *__vec)
     __attribute__ ((__nothrow__));
extern int shm_open (__const char *__name, int __oflag, mode_t __mode);


extern int shm_unlink (__const char *__name);

 typedef _Bool bool;
 typedef int8_t Int8;
  typedef int16_t Int16;
  typedef int32_t Int32;
  typedef uint8_t UInt8;
  typedef uint16_t UInt16;
  typedef uint32_t UInt32;






  typedef uint8_t Bool;
typedef char char_t;
typedef float Float;
typedef char Char;
typedef int Int;
typedef unsigned int UInt;
typedef char *String;





typedef char *Address;
typedef char const *ConstAddress;
typedef unsigned char Byte;
typedef float Float32;
typedef double Float64;
typedef void *Pointer;
typedef void const *ConstPointer;
typedef char const *ConstString;

typedef Int Endian;





typedef enum { TM32 = 0, TM32V2, TM64=100 } TMArch;
extern char* TMArch_names[];

typedef struct tmVersion
{
    UInt8 majorVersion;
    UInt8 minorVersion;
    UInt16 buildVersion;
} tmVersion_t, *ptmVersion_t;
typedef signed int IBits32;
typedef unsigned int UBits32;

typedef Int8 *pInt8;
typedef Int16 *pInt16;
typedef Int32 *pInt32;
typedef IBits32 *pIBits32;
typedef UBits32 *pUBits32;
typedef UInt8 *pUInt8;
typedef UInt16 *pUInt16;
typedef UInt32 *pUInt32;
typedef void Void, *pVoid;
typedef Float *pFloat;
typedef double Double, *pDouble;
typedef Bool *pBool;
typedef Char *pChar;
typedef Int *pInt;
typedef UInt *pUInt;
typedef String *pString;
typedef signed long long Int64, *pInt64;
typedef unsigned long long UInt64, *pUInt64;
typedef UInt32 tmErrorCode_t;
typedef UInt32 tmProgressCode_t;


typedef UInt64 tmTimeStamp_t, *ptmTimeStamp_t;





typedef union tmColor3
{
    UBits32 u32;
 struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 : 8;
    } rgb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 : 8;
    } yuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 : 8;
    } uml;

} tmColor3_t, *ptmColor3_t;

typedef union tmColor4
{
    UBits32 u32;
 struct {
        UBits32 blue : 8;
        UBits32 green : 8;
        UBits32 red : 8;
        UBits32 alpha : 8;
    } argb;
    struct {
        UBits32 v : 8;
        UBits32 u : 8;
        UBits32 y : 8;
        UBits32 alpha : 8;
    } ayuv;
    struct {
        UBits32 l : 8;
        UBits32 m : 8;
        UBits32 u : 8;
        UBits32 alpha : 8;
    } auml;

} tmColor4_t, *ptmColor4_t;




typedef enum tmPowerState
{
    tmPowerOn,
    tmPowerStandby,
    tmPowerSuspend,
    tmPowerOff

} tmPowerState_t, *ptmPowerState_t;




typedef struct tmSWVersion
{
    UInt32 compatibilityNr;
    UInt32 majorVersionNr;
    UInt32 minorVersionNr;

} tmSWVersion_t, *ptmSWVersion_t;
typedef Int tmUnitSelect_t, *ptmUnitSelect_t;
typedef Int tmInstance_t, *ptmInstance_t;


typedef Void (*ptmCallback_t) (UInt32 events, Void *pData, UInt32 userData);
typedef enum
{
    ph833xCore_M1,
    ph833xCore_Unknown
} ph833xCore_hwVersion;
extern tmErrorCode_t ph833xCore_setAudioRate( int audioRate );





extern tmErrorCode_t ph833xCore_getHwVersion( ph833xCore_hwVersion * pVersion );

extern tmErrorCode_t ph833xCore_Activate_Dsp(void);
typedef volatile struct {
     __u32 addr;
     __u32 stride;
     __u32 psize;
} PNXDrawSetupDestination_t;


typedef volatile struct {
     __u32 addr;
     __u32 stride;
} PNXDrawSetupSource_t;


typedef volatile struct {
     __u32 cccolor;
     __u32 transmask;
} PNXDrawSetupKey_t;


typedef volatile struct {
     __u32 bltctl;
     __u32 color;

     __u32 xy;
     __u32 size;
} PNXDrawExecuteFill_t;


typedef volatile struct {
     __u32 bltctl;

     __u32 srcxy;
     __u32 dstxy;
     __u32 size;
} PNXDrawExecuteBlit_t;


typedef volatile struct {
     __u32 bltctl;

     __u32 color;

     __u32 srcxy;
     __u32 dstxy;
     __u32 size;
} PNXDrawExecuteBlitBlend_t;


typedef volatile struct {
     __u32 header;
     union {
         PNXDrawSetupDestination_t dest;
         PNXDrawSetupSource_t source;
         PNXDrawSetupKey_t key;

         PNXDrawExecuteFill_t fill;
         PNXDrawExecuteBlit_t blit;
         PNXDrawExecuteBlitBlend_t blit_blend;
     } data;
} PNXDrawPacket_t;
typedef volatile struct {

    __u32 state;


    __u32 readIndex;

    __u32 writeIndex;

    PNXDrawPacket_t packets[170];
} PNXDrawSharedArea_t;
typedef struct __colour {
    uint32_t a;
    uint32_t r;
    uint32_t g;
    uint32_t b;
}colourInfo;


typedef struct __memArea {
    uint32_t physicalAddress;
    uint8_t * virtualAddress;
    int32_t size;
    uint32_t stride;
}memAreaInfo;
static int32_t gkeepCommandLoopAlive = 1;
static int32_t ghwAccEnabled = 0;
static int32_t gframeBufferDevice = 0;
static int32_t gDrawFD;
static int32_t width = 720;
static int32_t height = 576;

static colourInfo gcolourKey;

static memAreaInfo gmemRegion[4];

static PNXDrawSharedArea_t *gpDrawShared = ((void *)0);
static uint32_t hwAccNumPackets(void)
{
    uint32_t nrofPackets;
    if (gpDrawShared->writeIndex < gpDrawShared->readIndex)
    {
        nrofPackets = (170 - gpDrawShared->readIndex) + gpDrawShared->writeIndex;
    }
    else
    {
        nrofPackets = gpDrawShared->writeIndex - gpDrawShared->readIndex;
    }

    return nrofPackets;
}
static volatile void* hwAccPreparePacket(__u32 header)
{
    int32_t loop = 100;
    PNXDrawPacket_t *packet;

    while (hwAccNumPackets() == 170 - 1)
    {
        loop --;
        if (!loop)
        {
            (void)printf( "PNX8335: Timeout waiting for free packet entry, resetting!\n" );
            (void)ioctl( gDrawFD, (((0U) << (((0 +8)+8)+14)) | ((('D')) << (0 +8)) | (((0)) << 0) | ((0) << ((0 +8)+8))) );
        }

        (void)usleep( 10 );
    }
    packet = &gpDrawShared->packets[ gpDrawShared->writeIndex];
    packet->header = header;

    return &packet->data;
}






static void hwAccSubmitPacket(void)
{
    uint32_t wi = gpDrawShared->writeIndex;





    wi ++;
    if (wi >= 170)
    {
        wi = 0;
    }
    gpDrawShared->writeIndex = wi;

    if ((hwAccNumPackets() > (170/4)) &&
       !(gpDrawShared->state & 0x00000001))
    {
        (void)ioctl( gDrawFD, (((0U) << (((0 +8)+8)+14)) | ((('D')) << (0 +8)) | (((1)) << 0) | ((0) << ((0 +8)+8))));
    }
}






static int32_t hwAccInit(void)
{


    gDrawFD = open("/dev/pnxdraw", 02);
    if (gDrawFD >= 0)
    {

        gpDrawShared = mmap( 0, sizeof(PNXDrawSharedArea_t), 0x1 | 0x2, 0x01, gDrawFD, 0 );
        if (gpDrawShared == ((void *) -1))
        {
            (void)printf("Error mapping hw acc memory.\n");
            perror("/dev/pnxdraw");
            gpDrawShared = ((void *)0);
            (void)close( gDrawFD );
            return 1;
        }
    }
    else
    {
        (void)printf("Error opening hw acc device.\n");
        perror("/dev/pnxdraw");
        return 2;
    }

    return 0;
}






static void hwAccSync( void )
{
    if (gpDrawShared->state & 0x00000001)
    {
        (void)ioctl( gDrawFD, (((0U) << (((0 +8)+8)+14)) | ((('D')) << (0 +8)) | (((2)) << 0) | ((0) << ((0 +8)+8))) );
    }
}






static void hwAccFlush(void)
{
    while(hwAccNumPackets())
    {
        (void)ioctl( gDrawFD, (((0U) << (((0 +8)+8)+14)) | ((('D')) << (0 +8)) | (((1)) << 0) | ((0) << ((0 +8)+8))));
    }
}






static void hwAccDeinit(void)
{

    hwAccFlush();

    if (gpDrawShared)
    {

        (void)munmap( (void*)gpDrawShared, sizeof(PNXDrawSharedArea_t) );


        (void)close( gDrawFD );
    }
}






static void hwAccReset( void )
{
    (void)ioctl(gDrawFD, (((0U) << (((0 +8)+8)+14)) | ((('D')) << (0 +8)) | (((0)) << 0) | ((0) << ((0 +8)+8))) );
}
static uint32_t setupColour( int32_t format, colourInfo input )
{
    uint32_t colour = 0;
    switch (format)
    {
        case 0:
            colour = ( (((input.r)&0xF8) << 8) | (((input.g)&0xFC) << 3) | (((input.b)&0xF8) >> 3) );
            break;

        case 1:
            colour = ( (((input.a)&0xF0) << 8) | (((input.r)&0xF0) << 4) | (((input.g)&0xF0) ) | (((input.b)&0xF0) >> 4) );
            break;

        case 2:
            colour = ( ((input.a) << 24) | ((input.r) << 16) | ((input.g) << 8) | (input.b) );
            break;

        default:
            (void)printf( "unexpected pixelformat\n" );
    }
    return colour;
}
static void setupKey( colourInfo input )
{
    uint32_t colour;

    gcolourKey = input;
    colour = setupColour( 2, input );

    if (ghwAccEnabled)
    {
        PNXDrawSetupKey_t *keyPacket;
        keyPacket = hwAccPreparePacket(0x03);

        keyPacket->cccolor = colour;
        keyPacket->transmask = 0x00FFFFFF;

        hwAccSubmitPacket();
    }
}
static void setupSource( uint32_t src)
{
    PNXDrawSetupSource_t *srcPacket;

    srcPacket = hwAccPreparePacket(0x02);

    srcPacket->addr = gmemRegion[src].physicalAddress;
    srcPacket->stride = gmemRegion[src].stride;

    hwAccSubmitPacket();
}
static void setupDestination(uint32_t dest, int32_t format)
{
    PNXDrawSetupDestination_t *destPacket;

    destPacket = hwAccPreparePacket(0x01);
    destPacket->addr = gmemRegion[dest].physicalAddress;
    destPacket->stride = gmemRegion[dest].stride;

    switch (format)
    {
        case 0:
            destPacket->psize = 0x00000010 | 0x00000000;
            break;

        case 1:
            destPacket->psize = 0x00000010 | 0x00000100;
            break;

        case 2:
            destPacket->psize = 0x00000020 | 0x00000000;
            break;

        default:
            (void)printf( "unexpected pixelformat\n" );
    }

    hwAccSubmitPacket();
}
static void blitRegion( uint32_t src, uint32_t dest,
                        int32_t colourKeySrc, int32_t colourKeyDest,
                        int32_t x, int32_t y,
                        int32_t w, int32_t h,
                        int32_t destX, int32_t destY )
{
    if (ghwAccEnabled)
    {
        PNXDrawExecuteBlit_t *blit;

        setupSource(src);
        setupDestination(dest, 2);

        blit = hwAccPreparePacket(0x12);

        blit->bltctl = 0xCC;

        if (colourKeySrc)
        {
            blit->bltctl |= 0x00050000;
        }
        else
        {
            if (colourKeyDest)
            {
                blit->bltctl |= 0x00020000;
            }
        }

        blit->dstxy = (destX) | ((destY) << 16);
        blit->srcxy = (x) | ((y) << 16);
        blit->size = (w) | ((h) << 16);

        hwAccSubmitPacket();
    }
    else
    {
        int32_t i;
        int32_t j;
        uint8_t * srcPtr;
        uint8_t * destPtr;
        if (colourKeySrc)
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                for(j = 0; j < w; j++)
                {
                    if (((*(srcPtr)) != gcolourKey.b) ||
                        ((*((srcPtr)+1)) != gcolourKey.g) ||
                        ((*((srcPtr)+2)) != gcolourKey.r))
                    {
                        *destPtr = *srcPtr;
                        *(destPtr+1) = *(srcPtr+1);
                        *(destPtr+2) = *(srcPtr+2);
                        *(destPtr+3) = *(srcPtr+3);
                    }
                    srcPtr += 4;
                    destPtr += 4;
                }
            }
        }
        else
        if (colourKeyDest)
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                for(j = 0; j < w; j++)
                {
                    if (((*(destPtr)) == gcolourKey.b) &&
                        ((*((destPtr)+1)) == gcolourKey.g) &&
                        ((*((destPtr)+2)) == gcolourKey.r))

                    {
                        *destPtr = *srcPtr;
                        *(destPtr+1) = *(srcPtr+1);
                        *(destPtr+2) = *(srcPtr+2);
                        *(destPtr+3) = *(srcPtr+3);
                    }
                    srcPtr += 4;
                    destPtr += 4;
                }
            }
        }
        else
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                (void)memcpy( destPtr, srcPtr, (uint32_t)w*4);
            }
        }
    }
}
static void blendRegion( uint32_t src, uint32_t dest,
                         int32_t colourKeySrc, int32_t colourKeyDest,
                         int32_t x, int32_t y,
                         int32_t w, int32_t h,
                         int32_t destX, int32_t destY )
{
    if (ghwAccEnabled)
    {
        PNXDrawExecuteBlitBlend_t *blit;

        setupSource(src);
        setupDestination(dest, 2);

        blit = hwAccPreparePacket(0x13);

        blit->bltctl = 0x00400000 | 0x00100000;

        if (colourKeySrc)
        {
            blit->bltctl |= 0x00050000;
        }
        else
        {
            if (colourKeyDest)
            {
                blit->bltctl |= 0x00020000;
            }
        }

        blit->color = 0xFFFFFFFFu;
        blit->dstxy = (destX) | ((destY) << 16);
        blit->srcxy = (x) | ((y) << 16);
        blit->size = (w) | ((h) << 16);

        hwAccSubmitPacket();
    }
    else
    {
        int32_t i;
        int32_t j;
        uint8_t * srcPtr;
        uint8_t * destPtr;

        if (colourKeySrc)
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                for (j = 0; j < w; j++ )
                {
                    if (((*(srcPtr)) != gcolourKey.b) ||
                        ((*((srcPtr)+1)) != gcolourKey.g) ||
                        ((*((srcPtr)+2)) != gcolourKey.r))
                    {
                        uint32_t value;
                        uint32_t alpha = (uint32_t)(*((srcPtr)+3));
                        value = ((((uint32_t)(*((srcPtr)+2))) * alpha) + (((uint32_t)(*((destPtr)+2))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*((destPtr)+2)) = (uint8_t)value;
                        value = ((((uint32_t)(*((srcPtr)+1))) * alpha) + (((uint32_t)(*((destPtr)+1))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*((destPtr)+1)) = (uint8_t)value;
                        value = ((((uint32_t)(*(srcPtr))) * alpha) + (((uint32_t)(*(destPtr))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*(destPtr)) = (uint8_t)value;
                    }
                    destPtr += 4;
                    srcPtr += 4;
                }
            }
        }
        else
        if (colourKeyDest)
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                for (j = 0; j < w; j++ )
                {
                    if (((*(destPtr)) == gcolourKey.b) &&
                        ((*((destPtr)+1)) == gcolourKey.g) &&
                        ((*((destPtr)+2)) == gcolourKey.r))
                    {
                        uint32_t value;
                        uint32_t alpha = (uint32_t)(*((srcPtr)+3));
                        value = ((((uint32_t)(*((srcPtr)+2))) * alpha) + (((uint32_t)(*((destPtr)+2))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*((destPtr)+2)) = (uint8_t)value;
                        value = ((((uint32_t)(*((srcPtr)+1))) * alpha) + (((uint32_t)(*((destPtr)+1))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*((destPtr)+1)) = (uint8_t)value;
                        value = ((((uint32_t)(*(srcPtr))) * alpha) + (((uint32_t)(*(destPtr))) * (255-alpha)))/255;
                        if (value > 255)
                        {
                            value = 255;
                        }
                        (*(destPtr)) = (uint8_t)value;
                    }
                    destPtr += 4;
                    srcPtr += 4;
                }
            }
        }
        else
        {
            for (i = 0; i < h; i++ )
            {
                destPtr = gmemRegion[dest].virtualAddress + ((uint32_t)(destY+i)*gmemRegion[dest].stride) + (destX*4);
                srcPtr = gmemRegion[src].virtualAddress + ((uint32_t)(y+i)*gmemRegion[src].stride) + (x*4);
                for (j = 0; j < w; j++ )
                {
                    uint32_t value;
                    uint32_t alpha = (uint32_t)(*((srcPtr)+3));
                    value = alpha + ((uint32_t)(*((destPtr)+3)) * (255-alpha))/255;
                    if (value > 255)
                    {
                        value = 255;
                    }
                    (*((destPtr)+3)) = (uint8_t)value;
                    value = ((((uint32_t)(*((srcPtr)+2))) * alpha) + (((uint32_t)(*((destPtr)+2))) * (255-alpha)))/255;
                    if (value > 255)
                    {
                        value = 255;
                    }
                    (*((destPtr)+2)) = (uint8_t)value;
                    value = ((((uint32_t)(*((srcPtr)+1))) * alpha) + (((uint32_t)(*((destPtr)+1))) * (255-alpha)))/255;
                    if (value > 255)
                    {
                        value = 255;
                    }
                    (*((destPtr)+1)) = (uint8_t)value;
                    value = ((((uint32_t)(*(srcPtr))) * alpha) + (((uint32_t)(*(destPtr))) * (255-alpha)))/255;
                    if (value > 255)
                    {
                        value = 255;
                    }
                    (*(destPtr)) = (uint8_t)value;
                    destPtr += 4;
                    srcPtr += 4;
                }
            }
        }
    }
}
static void drawRectangle(uint32_t dest,
                          int32_t x, int32_t y,
                          int32_t w, int32_t h,
                          colourInfo input)
{
    if (ghwAccEnabled)
    {
        PNXDrawExecuteFill_t *fill;
        uint32_t colour;

        setupDestination(dest, 2);

        colour = setupColour( 2, input );

        fill = hwAccPreparePacket(0x11);

        fill->bltctl = 0x00001000 | 0xF0;
        fill->color = colour;

        fill->xy = (x) | ((y) << 16);
        fill->size = (w) | ((h) << 16);
        hwAccSubmitPacket();
    }
    else
    {
        int32_t i;
        int32_t j;

        for (i = 0; i < h; i++ )
        {
            uint8_t *pOutput = gmemRegion[dest].virtualAddress + ((uint32_t)(y+i)*gmemRegion[dest].stride) + (x*4);
            for (j = 0; j < w; j++ )
            {
                (*((pOutput)+3)) = (uint8_t)input.a;
                (*((pOutput)+2)) = (uint8_t)input.r;
                (*((pOutput)+1)) = (uint8_t)input.g;
                (*(pOutput)) = (uint8_t)input.b;
                pOutput += 4;
            }
        }
    }
}







static void clearDisplay(void)
{
    colourInfo input = {0};
    drawRectangle( 1, 0, 0, width, height, input);
}






static void updateDisplay(void)
{
    blitRegion( 1,
                0,
                0, 0,
                0, 0, width, height, 0, 0 );
    if (ghwAccEnabled)
    {
        hwAccFlush();
    }



}






static void createBlendedRegion(void)
{
    int32_t i;
    int32_t j;
    uint8_t *pOutput;
    pOutput = gmemRegion[2].virtualAddress;

    for (i = 0; i < 256; i++ )
    {
        for (j = 0; j < 256; j++ )
        {
            if ((i < 6) || (j < 6) || (i > 249) || (j > 249))
            {
                (*((pOutput)+3)) = 255;
                (*((pOutput)+2)) = 0;
                (*((pOutput)+1)) = 0;
                (*(pOutput)) = 0;
            }
            else
            {
                (*((pOutput)+3)) = (uint8_t)i;
                (*((pOutput)+2)) = 255;
                (*((pOutput)+1)) = 0;
                (*(pOutput)) = 0;
            }
            pOutput += 4;
        }
    }
}






static void creategcolourKeyRegion(void)
{
    colourInfo input;

    input.r = 0;
    input.g = 0;
    input.b = 0;
    input.a = 255;

    drawRectangle(3, 0, 0, 256, 256, input);

    input.r = 0;
    input.g = 0;
    input.b = 255;
    input.a = 255;

    drawRectangle(3, 256/4, 256/4,
                  256/2, 256/4, input);

    input.r = 255;
    input.g = 0;
    input.b = 255;
    input.a = 255;

    drawRectangle(3, 256/4, 256/2,
                  256/2, 256/4, input);
}
static void signalHandler(int32_t sig)
{
    static int32_t caught = 0;

    if (caught == 0)
    {
        caught = 1;
        (void)printf("Stopping Command Loop ... ((void)signal %d)\n", sig);
        gkeepCommandLoopAlive = 0;
    }
}
static int32_t openFrameBuffer(char * path)
{
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;


    gframeBufferDevice = open(path, 02);
    if (!gframeBufferDevice) {
        (void)printf("Error: cannot open framebuffer device %s\n", path);
        perror(path);
        return (1);
    }


    if (ioctl(gframeBufferDevice, 0x4602, &finfo)) {
        (void)printf("Error reading fixed information.\n");
        perror(path);
        return (2);
    }


    if (ioctl(gframeBufferDevice, 0x4600, &vinfo)) {
        (void)printf("Error reading variable information.\n");
        perror(path);
        return (3);
    }

    width = (int32_t)vinfo.xres;
    height = (int32_t)vinfo.yres;
    gmemRegion[0].size = (int32_t)(vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8);
    gmemRegion[0].stride = vinfo.xres*4;
    gmemRegion[1].size = (int32_t)(width*height*4);
    gmemRegion[1].stride = (uint32_t)(width*4);
    gmemRegion[2].size = (int32_t)(256*256*4);
    gmemRegion[2].stride = 256*4;
    gmemRegion[3].size = (int32_t)(256*256*4);
    gmemRegion[3].stride = 256*4;
    (void)printf("%d bytes allocated to the framebuffer\n", finfo.smem_len);
    (void)printf("%d bytes available after setting up a SINGLE buffered display\n", (int32_t)(finfo.smem_len) - gmemRegion[0].size);


    gmemRegion[0].physicalAddress = finfo.smem_start;
    gmemRegion[0].virtualAddress = (uint8_t *)mmap(0, (uint32_t)(gmemRegion[0].size +
                                                                     gmemRegion[1].size +
                                                                     gmemRegion[2].size +
                                                                     gmemRegion[3].size),
                                                                 0x1 | 0x2, 0x01,
                                                                 gframeBufferDevice, 0);
    if ((int32_t)gmemRegion[0].virtualAddress == -1) {
        (void)printf("Error: failed to map framebuffer device to memory.\n");
        perror(path);
        return (4);
    }


    gmemRegion[1].virtualAddress = gmemRegion[0].virtualAddress +
                                                          (uint32_t)gmemRegion[0].size;
    gmemRegion[1].physicalAddress = gmemRegion[0].physicalAddress +
                                                          (uint32_t)gmemRegion[0].size;


    gmemRegion[2].virtualAddress = gmemRegion[1].virtualAddress +
                                                      (uint32_t)gmemRegion[1].size;
    gmemRegion[2].physicalAddress = gmemRegion[1].physicalAddress +
                                                      (uint32_t)gmemRegion[1].size;


    gmemRegion[3].virtualAddress = gmemRegion[2].virtualAddress +
                                                           (uint32_t)gmemRegion[2].size;
    gmemRegion[3].physicalAddress = gmemRegion[2].physicalAddress +
                                                           (uint32_t)gmemRegion[2].size;

    return 0;
}






static void closeFrameBuffer(void)
{
    (void)munmap(gmemRegion[0].virtualAddress,
                (uint32_t)(gmemRegion[0].size +
                 gmemRegion[1].size +
                 gmemRegion[2].size +
                 gmemRegion[3].size));


    (void)close(gframeBufferDevice);
}
static int32_t randomValue(int32_t value)
{
    int32_t retValue;

    retValue = (int32_t)(((float)rand() * (float)value) / (float)2147483647);

    return retValue;
}





int32_t main(int32_t argc, char* argv[])
{
    int32_t status;
    int32_t i;
    int32_t j;
    colourInfo input;


__VERIFIER_assume(argc>=0 && argc<(sizeof(argv)/sizeof(char)));
int counter;
for(counter=0; counter<argc; counter++)
  __VERIFIER_assume(argv[counter]!=((void *)0));


    (void)printf("2D graphics hardware acceleration example.\n");
    if (argc > 1)
    {
        if (!strcmp( "-h", "-h"))
        {
            (void)printf("Usage : %s <-hw>\n", argv[0]);
            (void)printf("-hw     Enable hardware acceleration\n");
            exit(0);
        }
        ghwAccEnabled = !strcmp( "-hw", "-hw");
    }

    status = openFrameBuffer("/dev/fb0");


    if (status !=0)
    {
        exit(status);
    }

    status = hwAccInit();

    if (status !=0)
    {
        exit(status);
    }


    (void)signal(2, signalHandler);
    (void)signal(15, signalHandler);
    (void)signal(11, signalHandler);

    if (ghwAccEnabled)
    {
        hwAccReset();
        hwAccSync();
    }


    createBlendedRegion();


    creategcolourKeyRegion();




    for(i=0; gkeepCommandLoopAlive && (i<10); i++)
    {

        clearDisplay();

        for(j=0; gkeepCommandLoopAlive && (j<100); j++)
        {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;



            x = randomValue(width-2);
            y = randomValue(height-2);
            w = randomValue(width-x-1);
            if (w==0)
            {
                w = 1;
            }
            h = randomValue(height-y-1);
            if (h==0)
            {
                h = 1;
            }
            input.r = (uint32_t)randomValue(255);
            input.g = (uint32_t)randomValue(255);
            input.b = (uint32_t)randomValue(255);
            input.a = (uint32_t)randomValue(255);

            drawRectangle(1, x, y, w, h, input);
        }

        updateDisplay();
    }






    blitRegion( 2,
                1,
                0, 0,
                0, 0, 256, 256, 0, 0 );
    blitRegion( 2,
                1,
                0, 0,
                0, 0, 256, 256, 50, 50 );
    blitRegion( 2,
                1,
                0, 0,
                0, 0, 256, 256, 100, 100 );


    updateDisplay();






    blendRegion( 2,
                 1,
                 0, 0,
                 0, 0, 256, 256, 360, 0 );
    blendRegion( 2,
                 1,
                 0, 0,
                 0, 0, 256, 256, 360, 300 );

    updateDisplay();

    (void)sleep(2);





    input.r = 255;
    input.g = 0;
    input.b = 0;
    input.a = 255;

    drawRectangle(1, 0, 0, width/3, height, input);


    input.r = 0;
    input.g = 255;
    input.b = 0;
    input.a = 255;

    drawRectangle(1, width/3, 0, width/3, height, input);


    input.r = 0;
    input.g = 0;
    input.b = 255;
    input.a = 255;

    drawRectangle(1, (2*width)/3, 0, width/3, height, input);


    input.r = 0;
    input.g = 0;
    input.b = 255;
    input.a = 255;

    setupKey( input );

    blitRegion( 3,
                1,
                0, 0,
                0, 0, 256, 256, width/3 - 256/2, 40 );

    blitRegion( 3,
                1,
                1, 0,
                0, 0, 256, 256, 2*width/3 - 256/2, 40 );

    blitRegion( 3,
                1,
                0, 1,
                0, 0, 256, 256, 2*width/3 - 256/2, 300);

    updateDisplay();

    (void)closeFrameBuffer();

    hwAccDeinit();

    return 0;
}
