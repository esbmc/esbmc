# 1 "swarm_isort64.comb.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "swarm_isort64.comb.c"
# 1 "/usr/include/stdio.h" 1 3
# 28 "/usr/include/stdio.h" 3
# 1 "/usr/include/features.h" 1 3
# 335 "/usr/include/features.h" 3
# 1 "/usr/include/sys/cdefs.h" 1 3
# 360 "/usr/include/sys/cdefs.h" 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 361 "/usr/include/sys/cdefs.h" 2 3
# 336 "/usr/include/features.h" 2 3
# 359 "/usr/include/features.h" 3
# 1 "/usr/include/gnu/stubs.h" 1 3



# 1 "/usr/include/bits/wordsize.h" 1 3
# 5 "/usr/include/gnu/stubs.h" 2 3


# 1 "/usr/include/gnu/stubs-32.h" 1 3
# 8 "/usr/include/gnu/stubs.h" 2 3
# 360 "/usr/include/features.h" 2 3
# 29 "/usr/include/stdio.h" 2 3





# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 214 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 3 4
typedef unsigned int size_t;
# 35 "/usr/include/stdio.h" 2 3

# 1 "/usr/include/bits/types.h" 1 3
# 28 "/usr/include/bits/types.h" 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 29 "/usr/include/bits/types.h" 2 3


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
# 131 "/usr/include/bits/types.h" 3
# 1 "/usr/include/bits/typesizes.h" 1 3
# 132 "/usr/include/bits/types.h" 2 3


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
# 37 "/usr/include/stdio.h" 2 3
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;



typedef struct _IO_FILE FILE;





# 65 "/usr/include/stdio.h" 3
typedef struct _IO_FILE __FILE;
# 75 "/usr/include/stdio.h" 3
# 1 "/usr/include/libio.h" 1 3
# 32 "/usr/include/libio.h" 3
# 1 "/usr/include/_G_config.h" 1 3
# 15 "/usr/include/_G_config.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 16 "/usr/include/_G_config.h" 2 3




# 1 "/usr/include/wchar.h" 1 3
# 78 "/usr/include/wchar.h" 3
typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;
# 21 "/usr/include/_G_config.h" 2 3

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
# 53 "/usr/include/_G_config.h" 3
typedef int _G_int16_t __attribute__ ((__mode__ (__HI__)));
typedef int _G_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int _G_uint16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int _G_uint32_t __attribute__ ((__mode__ (__SI__)));
# 33 "/usr/include/libio.h" 2 3
# 53 "/usr/include/libio.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 1 3 4
# 43 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 54 "/usr/include/libio.h" 2 3
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
# 203 "/usr/include/libio.h" 3
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
# 271 "/usr/include/libio.h" 3
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
# 319 "/usr/include/libio.h" 3
  __off64_t _offset;
# 328 "/usr/include/libio.h" 3
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
# 364 "/usr/include/libio.h" 3
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, __const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);
# 416 "/usr/include/libio.h" 3
extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
# 458 "/usr/include/libio.h" 3
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __attribute__ ((__nothrow__));
extern int _IO_ferror (_IO_FILE *__fp) __attribute__ ((__nothrow__));

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern void _IO_funlockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern int _IO_ftrylockfile (_IO_FILE *) __attribute__ ((__nothrow__));
# 488 "/usr/include/libio.h" 3
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __attribute__ ((__nothrow__));
# 76 "/usr/include/stdio.h" 2 3
# 89 "/usr/include/stdio.h" 3


typedef _G_fpos_t fpos_t;




# 141 "/usr/include/stdio.h" 3
# 1 "/usr/include/bits/stdio_lim.h" 1 3
# 142 "/usr/include/stdio.h" 2 3



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;









extern int remove (__const char *__filename) __attribute__ ((__nothrow__));

extern int rename (__const char *__old, __const char *__new) __attribute__ ((__nothrow__));














extern FILE *tmpfile (void) ;
# 188 "/usr/include/stdio.h" 3
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__)) ;





extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__)) ;
# 206 "/usr/include/stdio.h" 3
extern char *tempnam (__const char *__dir, __const char *__pfx)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;








extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);

# 231 "/usr/include/stdio.h" 3
extern int fflush_unlocked (FILE *__stream);
# 245 "/usr/include/stdio.h" 3






extern FILE *fopen (__const char *__restrict __filename,
      __const char *__restrict __modes) ;




extern FILE *freopen (__const char *__restrict __filename,
        __const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 274 "/usr/include/stdio.h" 3

# 285 "/usr/include/stdio.h" 3
extern FILE *fdopen (int __fd, __const char *__modes) __attribute__ ((__nothrow__)) ;
# 306 "/usr/include/stdio.h" 3



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

# 400 "/usr/include/stdio.h" 3





extern int fscanf (FILE *__restrict __stream,
     __const char *__restrict __format, ...) ;




extern int scanf (__const char *__restrict __format, ...) ;

extern int sscanf (__const char *__restrict __s,
     __const char *__restrict __format, ...) __attribute__ ((__nothrow__));
# 443 "/usr/include/stdio.h" 3

# 506 "/usr/include/stdio.h" 3





extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);

# 530 "/usr/include/stdio.h" 3
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 541 "/usr/include/stdio.h" 3
extern int fgetc_unlocked (FILE *__stream);











extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);

# 574 "/usr/include/stdio.h" 3
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);








extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;






extern char *gets (char *__s) ;

# 655 "/usr/include/stdio.h" 3





extern int fputs (__const char *__restrict __s, FILE *__restrict __stream);





extern int puts (__const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (__const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s) ;

# 708 "/usr/include/stdio.h" 3
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (__const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream) ;








extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);

# 744 "/usr/include/stdio.h" 3
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 763 "/usr/include/stdio.h" 3






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, __const fpos_t *__pos);
# 786 "/usr/include/stdio.h" 3

# 795 "/usr/include/stdio.h" 3


extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__)) ;

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;








extern void perror (__const char *__s);






# 1 "/usr/include/bits/sys_errlist.h" 1 3
# 27 "/usr/include/bits/sys_errlist.h" 3
extern int sys_nerr;
extern __const char *__const sys_errlist[];
# 825 "/usr/include/stdio.h" 2 3




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
# 844 "/usr/include/stdio.h" 3
extern FILE *popen (__const char *__command, __const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__));
# 884 "/usr/include/stdio.h" 3
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__)) ;


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__));
# 914 "/usr/include/stdio.h" 3

# 2 "swarm_isort64.comb.c" 2
# 1 "/usr/include/time.h" 1 3
# 31 "/usr/include/time.h" 3








# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 40 "/usr/include/time.h" 2 3



# 1 "/usr/include/bits/time.h" 1 3
# 44 "/usr/include/time.h" 2 3
# 59 "/usr/include/time.h" 3


typedef __clock_t clock_t;



# 75 "/usr/include/time.h" 3


typedef __time_t time_t;



# 93 "/usr/include/time.h" 3
typedef __clockid_t clockid_t;
# 105 "/usr/include/time.h" 3
typedef __timer_t timer_t;
# 121 "/usr/include/time.h" 3
struct timespec
  {
    __time_t tv_sec;
    long int tv_nsec;
  };








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
  __const char *tm_zone;




};








struct itimerspec
  {
    struct timespec it_interval;
    struct timespec it_value;
  };


struct sigevent;





typedef __pid_t pid_t;








extern clock_t clock (void) __attribute__ ((__nothrow__));


extern time_t time (time_t *__timer) __attribute__ ((__nothrow__));


extern double difftime (time_t __time1, time_t __time0)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern time_t mktime (struct tm *__tp) __attribute__ ((__nothrow__));





extern size_t strftime (char *__restrict __s, size_t __maxsize,
   __const char *__restrict __format,
   __const struct tm *__restrict __tp) __attribute__ ((__nothrow__));

# 229 "/usr/include/time.h" 3



extern struct tm *gmtime (__const time_t *__timer) __attribute__ ((__nothrow__));



extern struct tm *localtime (__const time_t *__timer) __attribute__ ((__nothrow__));





extern struct tm *gmtime_r (__const time_t *__restrict __timer,
       struct tm *__restrict __tp) __attribute__ ((__nothrow__));



extern struct tm *localtime_r (__const time_t *__restrict __timer,
          struct tm *__restrict __tp) __attribute__ ((__nothrow__));





extern char *asctime (__const struct tm *__tp) __attribute__ ((__nothrow__));


extern char *ctime (__const time_t *__timer) __attribute__ ((__nothrow__));







extern char *asctime_r (__const struct tm *__restrict __tp,
   char *__restrict __buf) __attribute__ ((__nothrow__));


extern char *ctime_r (__const time_t *__restrict __timer,
        char *__restrict __buf) __attribute__ ((__nothrow__));




extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;




extern char *tzname[2];



extern void tzset (void) __attribute__ ((__nothrow__));



extern int daylight;
extern long int timezone;





extern int stime (__const time_t *__when) __attribute__ ((__nothrow__));
# 312 "/usr/include/time.h" 3
extern time_t timegm (struct tm *__tp) __attribute__ ((__nothrow__));


extern time_t timelocal (struct tm *__tp) __attribute__ ((__nothrow__));


extern int dysize (int __year) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 327 "/usr/include/time.h" 3
extern int nanosleep (__const struct timespec *__requested_time,
        struct timespec *__remaining);



extern int clock_getres (clockid_t __clock_id, struct timespec *__res) __attribute__ ((__nothrow__));


extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) __attribute__ ((__nothrow__));


extern int clock_settime (clockid_t __clock_id, __const struct timespec *__tp)
     __attribute__ ((__nothrow__));






extern int clock_nanosleep (clockid_t __clock_id, int __flags,
       __const struct timespec *__req,
       struct timespec *__rem);


extern int clock_getcpuclockid (pid_t __pid, clockid_t *__clock_id) __attribute__ ((__nothrow__));




extern int timer_create (clockid_t __clock_id,
    struct sigevent *__restrict __evp,
    timer_t *__restrict __timerid) __attribute__ ((__nothrow__));


extern int timer_delete (timer_t __timerid) __attribute__ ((__nothrow__));


extern int timer_settime (timer_t __timerid, int __flags,
     __const struct itimerspec *__restrict __value,
     struct itimerspec *__restrict __ovalue) __attribute__ ((__nothrow__));


extern int timer_gettime (timer_t __timerid, struct itimerspec *__value)
     __attribute__ ((__nothrow__));


extern int timer_getoverrun (timer_t __timerid) __attribute__ ((__nothrow__));
# 416 "/usr/include/time.h" 3

# 3 "swarm_isort64.comb.c" 2
# 1 "/usr/include/stdlib.h" 1 3
# 33 "/usr/include/stdlib.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 326 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 3 4
typedef long int wchar_t;
# 34 "/usr/include/stdlib.h" 2 3


# 96 "/usr/include/stdlib.h" 3


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



# 140 "/usr/include/stdlib.h" 3
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

# 182 "/usr/include/stdlib.h" 3


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

# 311 "/usr/include/stdlib.h" 3
extern char *l64a (long int __n) __attribute__ ((__nothrow__)) ;


extern long int a64l (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;




# 1 "/usr/include/sys/types.h" 1 3
# 29 "/usr/include/sys/types.h" 3






typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;




typedef __loff_t loff_t;



typedef __ino_t ino_t;
# 62 "/usr/include/sys/types.h" 3
typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;





typedef __off_t off_t;
# 105 "/usr/include/sys/types.h" 3
typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 147 "/usr/include/sys/types.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 148 "/usr/include/sys/types.h" 2 3



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
# 195 "/usr/include/sys/types.h" 3
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));
# 217 "/usr/include/sys/types.h" 3
# 1 "/usr/include/endian.h" 1 3
# 37 "/usr/include/endian.h" 3
# 1 "/usr/include/bits/endian.h" 1 3
# 38 "/usr/include/endian.h" 2 3
# 61 "/usr/include/endian.h" 3
# 1 "/usr/include/bits/byteswap.h" 1 3
# 62 "/usr/include/endian.h" 2 3
# 218 "/usr/include/sys/types.h" 2 3


# 1 "/usr/include/sys/select.h" 1 3
# 31 "/usr/include/sys/select.h" 3
# 1 "/usr/include/bits/select.h" 1 3
# 32 "/usr/include/sys/select.h" 2 3


# 1 "/usr/include/bits/sigset.h" 1 3
# 24 "/usr/include/bits/sigset.h" 3
typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;
# 35 "/usr/include/sys/select.h" 2 3



typedef __sigset_t sigset_t;







# 1 "/usr/include/bits/time.h" 1 3
# 69 "/usr/include/bits/time.h" 3
struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };
# 47 "/usr/include/sys/select.h" 2 3


typedef __suseconds_t suseconds_t;





typedef long int __fd_mask;
# 67 "/usr/include/sys/select.h" 3
typedef struct
  {






    __fd_mask __fds_bits[1024 / (8 * sizeof (__fd_mask))];


  } fd_set;






typedef __fd_mask fd_mask;
# 99 "/usr/include/sys/select.h" 3

# 109 "/usr/include/sys/select.h" 3
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 121 "/usr/include/sys/select.h" 3
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);



# 221 "/usr/include/sys/types.h" 2 3


# 1 "/usr/include/sys/sysmacros.h" 1 3
# 30 "/usr/include/sys/sysmacros.h" 3
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
# 224 "/usr/include/sys/types.h" 2 3
# 235 "/usr/include/sys/types.h" 3
typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 270 "/usr/include/sys/types.h" 3
# 1 "/usr/include/bits/pthreadtypes.h" 1 3
# 36 "/usr/include/bits/pthreadtypes.h" 3
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
# 271 "/usr/include/sys/types.h" 2 3



# 321 "/usr/include/stdlib.h" 2 3






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



# 1 "/usr/include/alloca.h" 1 3
# 25 "/usr/include/alloca.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 26 "/usr/include/alloca.h" 2 3







extern void *alloca (size_t __size) __attribute__ ((__nothrow__));






# 498 "/usr/include/stdlib.h" 2 3




extern void *valloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern void abort (void) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));

# 543 "/usr/include/stdlib.h" 3


extern char *getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern char *__secure_getenv (__const char *__name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int putenv (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (__const char *__name, __const char *__value, int __replace)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (__const char *__name) __attribute__ ((__nothrow__));






extern int clearenv (void) __attribute__ ((__nothrow__));
# 583 "/usr/include/stdlib.h" 3
extern char *mktemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 594 "/usr/include/stdlib.h" 3
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 614 "/usr/include/stdlib.h" 3
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 640 "/usr/include/stdlib.h" 3





extern int system (__const char *__command) ;

# 662 "/usr/include/stdlib.h" 3
extern char *realpath (__const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__)) ;






typedef int (*__compar_fn_t) (__const void *, __const void *);
# 680 "/usr/include/stdlib.h" 3



extern void *bsearch (__const void *__key, __const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;



extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
# 699 "/usr/include/stdlib.h" 3
extern int abs (int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;












extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;

# 735 "/usr/include/stdlib.h" 3
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
# 840 "/usr/include/stdlib.h" 3
extern int posix_openpt (int __oflag) ;
# 875 "/usr/include/stdlib.h" 3
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 891 "/usr/include/stdlib.h" 3

# 4 "swarm_isort64.comb.c" 2
# 1 "/usr/include/pthread.h" 1 3
# 25 "/usr/include/pthread.h" 3
# 1 "/usr/include/sched.h" 1 3
# 29 "/usr/include/sched.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 30 "/usr/include/sched.h" 2 3





# 1 "/usr/include/bits/sched.h" 1 3
# 71 "/usr/include/bits/sched.h" 3
struct sched_param
  {
    int __sched_priority;
  };





extern int clone (int (*__fn) (void *__arg), void *__child_stack,
    int __flags, void *__arg, ...) __attribute__ ((__nothrow__));


extern int unshare (int __flags) __attribute__ ((__nothrow__));


extern int sched_getcpu (void) __attribute__ ((__nothrow__));










struct __sched_param
  {
    int __sched_priority;
  };
# 113 "/usr/include/bits/sched.h" 3
typedef unsigned long int __cpu_mask;






typedef struct
{
  __cpu_mask __bits[1024 / (8 * sizeof (__cpu_mask))];
} cpu_set_t;
# 196 "/usr/include/bits/sched.h" 3


extern int __sched_cpucount (size_t __setsize, const cpu_set_t *__setp)
  __attribute__ ((__nothrow__));
extern cpu_set_t *__sched_cpualloc (size_t __count) __attribute__ ((__nothrow__)) ;
extern void __sched_cpufree (cpu_set_t *__set) __attribute__ ((__nothrow__));


# 36 "/usr/include/sched.h" 2 3







extern int sched_setparam (__pid_t __pid, __const struct sched_param *__param)
     __attribute__ ((__nothrow__));


extern int sched_getparam (__pid_t __pid, struct sched_param *__param) __attribute__ ((__nothrow__));


extern int sched_setscheduler (__pid_t __pid, int __policy,
          __const struct sched_param *__param) __attribute__ ((__nothrow__));


extern int sched_getscheduler (__pid_t __pid) __attribute__ ((__nothrow__));


extern int sched_yield (void) __attribute__ ((__nothrow__));


extern int sched_get_priority_max (int __algorithm) __attribute__ ((__nothrow__));


extern int sched_get_priority_min (int __algorithm) __attribute__ ((__nothrow__));


extern int sched_rr_get_interval (__pid_t __pid, struct timespec *__t) __attribute__ ((__nothrow__));
# 118 "/usr/include/sched.h" 3

# 26 "/usr/include/pthread.h" 2 3



# 1 "/usr/include/signal.h" 1 3
# 31 "/usr/include/signal.h" 3


# 1 "/usr/include/bits/sigset.h" 1 3
# 34 "/usr/include/signal.h" 2 3
# 402 "/usr/include/signal.h" 3

# 30 "/usr/include/pthread.h" 2 3

# 1 "/usr/include/bits/setjmp.h" 1 3
# 29 "/usr/include/bits/setjmp.h" 3
typedef int __jmp_buf[6];
# 32 "/usr/include/pthread.h" 2 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 33 "/usr/include/pthread.h" 2 3



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
# 63 "/usr/include/pthread.h" 3
};
# 115 "/usr/include/pthread.h" 3
enum
{
  PTHREAD_RWLOCK_PREFER_READER_NP,
  PTHREAD_RWLOCK_PREFER_WRITER_NP,
  PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
  PTHREAD_RWLOCK_DEFAULT_NP = PTHREAD_RWLOCK_PREFER_READER_NP
};
# 147 "/usr/include/pthread.h" 3
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
# 182 "/usr/include/pthread.h" 3
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
# 220 "/usr/include/pthread.h" 3





extern int pthread_create (pthread_t *__restrict __newthread,
      __const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));





extern void pthread_exit (void *__retval) __attribute__ ((__noreturn__));







extern int pthread_join (pthread_t __th, void **__thread_return);
# 263 "/usr/include/pthread.h" 3
extern int pthread_detach (pthread_t __th) __attribute__ ((__nothrow__));



extern pthread_t pthread_self (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int pthread_equal (pthread_t __thread1, pthread_t __thread2) __attribute__ ((__nothrow__));







extern int pthread_attr_init (pthread_attr_t *__attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_destroy (pthread_attr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getdetachstate (__const pthread_attr_t *__attr,
     int *__detachstate)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setdetachstate (pthread_attr_t *__attr,
     int __detachstate)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getguardsize (__const pthread_attr_t *__attr,
          size_t *__guardsize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setguardsize (pthread_attr_t *__attr,
          size_t __guardsize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getschedparam (__const pthread_attr_t *__restrict
           __attr,
           struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setschedparam (pthread_attr_t *__restrict __attr,
           __const struct sched_param *__restrict
           __param) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_getschedpolicy (__const pthread_attr_t *__restrict
     __attr, int *__restrict __policy)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setschedpolicy (pthread_attr_t *__attr, int __policy)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getinheritsched (__const pthread_attr_t *__restrict
      __attr, int *__restrict __inherit)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setinheritsched (pthread_attr_t *__attr,
      int __inherit)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getscope (__const pthread_attr_t *__restrict __attr,
      int *__restrict __scope)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_attr_setscope (pthread_attr_t *__attr, int __scope)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_attr_getstackaddr (__const pthread_attr_t *__restrict
          __attr, void **__restrict __stackaddr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) __attribute__ ((__deprecated__));





extern int pthread_attr_setstackaddr (pthread_attr_t *__attr,
          void *__stackaddr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__));


extern int pthread_attr_getstacksize (__const pthread_attr_t *__restrict
          __attr, size_t *__restrict __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int pthread_attr_setstacksize (pthread_attr_t *__attr,
          size_t __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_attr_getstack (__const pthread_attr_t *__restrict __attr,
      void **__restrict __stackaddr,
      size_t *__restrict __stacksize)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2, 3)));




extern int pthread_attr_setstack (pthread_attr_t *__attr, void *__stackaddr,
      size_t __stacksize) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 413 "/usr/include/pthread.h" 3
extern int pthread_setschedparam (pthread_t __target_thread, int __policy,
      __const struct sched_param *__param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));


extern int pthread_getschedparam (pthread_t __target_thread,
      int *__restrict __policy,
      struct sched_param *__restrict __param)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3)));


extern int pthread_setschedprio (pthread_t __target_thread, int __prio)
     __attribute__ ((__nothrow__));
# 466 "/usr/include/pthread.h" 3
extern int pthread_once (pthread_once_t *__once_control,
    void (*__init_routine) (void)) __attribute__ ((__nonnull__ (1, 2)));
# 478 "/usr/include/pthread.h" 3
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
# 512 "/usr/include/pthread.h" 3
struct __pthread_cleanup_frame
{
  void (*__cancel_routine) (void *);
  void *__cancel_arg;
  int __do_it;
  int __cancel_type;
};
# 652 "/usr/include/pthread.h" 3
extern void __pthread_register_cancel (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1)));
# 664 "/usr/include/pthread.h" 3
extern void __pthread_unregister_cancel (__pthread_unwind_buf_t *__buf)
  __attribute__ ((__regparm__ (1)));
# 705 "/usr/include/pthread.h" 3
extern void __pthread_unwind_next (__pthread_unwind_buf_t *__buf)
     __attribute__ ((__regparm__ (1))) __attribute__ ((__noreturn__))

     __attribute__ ((__weak__))

     ;



struct __jmp_buf_tag;
extern int __sigsetjmp (struct __jmp_buf_tag *__env, int __savemask) __attribute__ ((__nothrow__));





extern int pthread_mutex_init (pthread_mutex_t *__mutex,
          __const pthread_mutexattr_t *__mutexattr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_destroy (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_trylock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutex_lock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_mutex_timedlock (pthread_mutex_t *__restrict __mutex,
                                    __const struct timespec *__restrict
                                    __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_mutex_unlock (pthread_mutex_t *__mutex)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 776 "/usr/include/pthread.h" 3
extern int pthread_mutexattr_init (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutexattr_destroy (pthread_mutexattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_mutexattr_getpshared (__const pthread_mutexattr_t *
      __restrict __attr,
      int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_mutexattr_setpshared (pthread_mutexattr_t *__attr,
      int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 848 "/usr/include/pthread.h" 3
extern int pthread_rwlock_init (pthread_rwlock_t *__restrict __rwlock,
    __const pthread_rwlockattr_t *__restrict
    __attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_destroy (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_rdlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_tryrdlock (pthread_rwlock_t *__rwlock)
  __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_rwlock_timedrdlock (pthread_rwlock_t *__restrict __rwlock,
           __const struct timespec *__restrict
           __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_rwlock_wrlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlock_trywrlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_rwlock_timedwrlock (pthread_rwlock_t *__restrict __rwlock,
           __const struct timespec *__restrict
           __abstime) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int pthread_rwlock_unlock (pthread_rwlock_t *__rwlock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int pthread_rwlockattr_init (pthread_rwlockattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_destroy (pthread_rwlockattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_getpshared (__const pthread_rwlockattr_t *
       __restrict __attr,
       int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_rwlockattr_setpshared (pthread_rwlockattr_t *__attr,
       int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_rwlockattr_getkind_np (__const pthread_rwlockattr_t *
       __restrict __attr,
       int *__restrict __pref)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_rwlockattr_setkind_np (pthread_rwlockattr_t *__attr,
       int __pref) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







extern int pthread_cond_init (pthread_cond_t *__restrict __cond,
         __const pthread_condattr_t *__restrict
         __cond_attr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_destroy (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_signal (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_cond_broadcast (pthread_cond_t *__cond)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int pthread_cond_wait (pthread_cond_t *__restrict __cond,
         pthread_mutex_t *__restrict __mutex)
     __attribute__ ((__nonnull__ (1, 2)));
# 960 "/usr/include/pthread.h" 3
extern int pthread_cond_timedwait (pthread_cond_t *__restrict __cond,
       pthread_mutex_t *__restrict __mutex,
       __const struct timespec *__restrict
       __abstime) __attribute__ ((__nonnull__ (1, 2, 3)));




extern int pthread_condattr_init (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_condattr_destroy (pthread_condattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_condattr_getpshared (__const pthread_condattr_t *
                                        __restrict __attr,
                                        int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_condattr_setpshared (pthread_condattr_t *__attr,
                                        int __pshared) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_condattr_getclock (__const pthread_condattr_t *
          __restrict __attr,
          __clockid_t *__restrict __clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_condattr_setclock (pthread_condattr_t *__attr,
          __clockid_t __clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 1004 "/usr/include/pthread.h" 3
extern int pthread_spin_init (pthread_spinlock_t *__lock, int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_destroy (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_lock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_trylock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_spin_unlock (pthread_spinlock_t *__lock)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int pthread_barrier_init (pthread_barrier_t *__restrict __barrier,
     __const pthread_barrierattr_t *__restrict
     __attr, unsigned int __count)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrier_destroy (pthread_barrier_t *__barrier)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrier_wait (pthread_barrier_t *__barrier)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int pthread_barrierattr_init (pthread_barrierattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrierattr_destroy (pthread_barrierattr_t *__attr)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_barrierattr_getpshared (__const pthread_barrierattr_t *
        __restrict __attr,
        int *__restrict __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern int pthread_barrierattr_setpshared (pthread_barrierattr_t *__attr,
                                           int __pshared)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 1071 "/usr/include/pthread.h" 3
extern int pthread_key_create (pthread_key_t *__key,
          void (*__destr_function) (void *))
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int pthread_key_delete (pthread_key_t __key) __attribute__ ((__nothrow__));


extern void *pthread_getspecific (pthread_key_t __key) __attribute__ ((__nothrow__));


extern int pthread_setspecific (pthread_key_t __key,
    __const void *__pointer) __attribute__ ((__nothrow__)) ;




extern int pthread_getcpuclockid (pthread_t __thread_id,
      __clockid_t *__clock_id)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
# 1105 "/usr/include/pthread.h" 3
extern int pthread_atfork (void (*__prepare) (void),
      void (*__parent) (void),
      void (*__child) (void)) __attribute__ ((__nothrow__));
# 1119 "/usr/include/pthread.h" 3

# 5 "swarm_isort64.comb.c" 2
# 1 "/usr/include/string.h" 1 3
# 28 "/usr/include/string.h" 3





# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 34 "/usr/include/string.h" 2 3




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

# 82 "/usr/include/string.h" 3


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

# 130 "/usr/include/string.h" 3
extern char *strdup (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));
# 165 "/usr/include/string.h" 3


extern char *strchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern char *strrchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

# 181 "/usr/include/string.h" 3



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
# 240 "/usr/include/string.h" 3


extern size_t strlen (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

# 254 "/usr/include/string.h" 3


extern char *strerror (int __errnum) __attribute__ ((__nothrow__));

# 270 "/usr/include/string.h" 3
extern int strerror_r (int __errnum, char *__buf, size_t __buflen) __asm__ ("" "__xpg_strerror_r") __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));
# 294 "/usr/include/string.h" 3
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
# 331 "/usr/include/string.h" 3
extern int strcasecmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 354 "/usr/include/string.h" 3
extern char *strsep (char **__restrict __stringp,
       __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
# 432 "/usr/include/string.h" 3

# 6 "swarm_isort64.comb.c" 2
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 1 3 4
# 105 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdarg.h" 3 4
typedef __gnuc_va_list va_list;
# 7 "swarm_isort64.comb.c" 2
# 1 "/usr/include/errno.h" 1 3
# 32 "/usr/include/errno.h" 3




# 1 "/usr/include/bits/errno.h" 1 3
# 25 "/usr/include/bits/errno.h" 3
# 1 "/usr/include/linux/errno.h" 1 3



# 1 "/usr/include/asm/errno.h" 1 3



# 1 "/usr/include/asm-generic/errno.h" 1 3



# 1 "/usr/include/asm-generic/errno-base.h" 1 3
# 5 "/usr/include/asm-generic/errno.h" 2 3
# 5 "/usr/include/asm/errno.h" 2 3
# 5 "/usr/include/linux/errno.h" 2 3
# 26 "/usr/include/bits/errno.h" 2 3
# 43 "/usr/include/bits/errno.h" 3
extern int *__errno_location (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 37 "/usr/include/errno.h" 2 3
# 59 "/usr/include/errno.h" 3

# 8 "swarm_isort64.comb.c" 2
# 1 "/usr/include/ctype.h" 1 3
# 30 "/usr/include/ctype.h" 3

# 48 "/usr/include/ctype.h" 3
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
# 81 "/usr/include/ctype.h" 3
extern __const unsigned short int **__ctype_b_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_tolower_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_toupper_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
# 96 "/usr/include/ctype.h" 3






extern int isalnum (int) __attribute__ ((__nothrow__));
extern int isalpha (int) __attribute__ ((__nothrow__));
extern int iscntrl (int) __attribute__ ((__nothrow__));
extern int isdigit (int) __attribute__ ((__nothrow__));
extern int islower (int) __attribute__ ((__nothrow__));
extern int isgraph (int) __attribute__ ((__nothrow__));
extern int isprint (int) __attribute__ ((__nothrow__));
extern int ispunct (int) __attribute__ ((__nothrow__));
extern int isspace (int) __attribute__ ((__nothrow__));
extern int isupper (int) __attribute__ ((__nothrow__));
extern int isxdigit (int) __attribute__ ((__nothrow__));



extern int tolower (int __c) __attribute__ ((__nothrow__));


extern int toupper (int __c) __attribute__ ((__nothrow__));


# 142 "/usr/include/ctype.h" 3
extern int isascii (int __c) __attribute__ ((__nothrow__));



extern int toascii (int __c) __attribute__ ((__nothrow__));



extern int _toupper (int) __attribute__ ((__nothrow__));
extern int _tolower (int) __attribute__ ((__nothrow__));
# 323 "/usr/include/ctype.h" 3

# 9 "swarm_isort64.comb.c" 2

# 1 "/usr/include/math.h" 1 3
# 30 "/usr/include/math.h" 3




# 1 "/usr/include/bits/huge_val.h" 1 3
# 35 "/usr/include/math.h" 2 3
# 47 "/usr/include/math.h" 3
# 1 "/usr/include/bits/mathdef.h" 1 3
# 48 "/usr/include/math.h" 2 3
# 71 "/usr/include/math.h" 3
# 1 "/usr/include/bits/mathcalls.h" 1 3
# 53 "/usr/include/bits/mathcalls.h" 3


extern double acos (double __x) __attribute__ ((__nothrow__)); extern double __acos (double __x) __attribute__ ((__nothrow__));

extern double asin (double __x) __attribute__ ((__nothrow__)); extern double __asin (double __x) __attribute__ ((__nothrow__));

extern double atan (double __x) __attribute__ ((__nothrow__)); extern double __atan (double __x) __attribute__ ((__nothrow__));

extern double atan2 (double __y, double __x) __attribute__ ((__nothrow__)); extern double __atan2 (double __y, double __x) __attribute__ ((__nothrow__));


extern double cos (double __x) __attribute__ ((__nothrow__)); extern double __cos (double __x) __attribute__ ((__nothrow__));

extern double sin (double __x) __attribute__ ((__nothrow__)); extern double __sin (double __x) __attribute__ ((__nothrow__));

extern double tan (double __x) __attribute__ ((__nothrow__)); extern double __tan (double __x) __attribute__ ((__nothrow__));




extern double cosh (double __x) __attribute__ ((__nothrow__)); extern double __cosh (double __x) __attribute__ ((__nothrow__));

extern double sinh (double __x) __attribute__ ((__nothrow__)); extern double __sinh (double __x) __attribute__ ((__nothrow__));

extern double tanh (double __x) __attribute__ ((__nothrow__)); extern double __tanh (double __x) __attribute__ ((__nothrow__));

# 87 "/usr/include/bits/mathcalls.h" 3


extern double acosh (double __x) __attribute__ ((__nothrow__)); extern double __acosh (double __x) __attribute__ ((__nothrow__));

extern double asinh (double __x) __attribute__ ((__nothrow__)); extern double __asinh (double __x) __attribute__ ((__nothrow__));

extern double atanh (double __x) __attribute__ ((__nothrow__)); extern double __atanh (double __x) __attribute__ ((__nothrow__));







extern double exp (double __x) __attribute__ ((__nothrow__)); extern double __exp (double __x) __attribute__ ((__nothrow__));


extern double frexp (double __x, int *__exponent) __attribute__ ((__nothrow__)); extern double __frexp (double __x, int *__exponent) __attribute__ ((__nothrow__));


extern double ldexp (double __x, int __exponent) __attribute__ ((__nothrow__)); extern double __ldexp (double __x, int __exponent) __attribute__ ((__nothrow__));


extern double log (double __x) __attribute__ ((__nothrow__)); extern double __log (double __x) __attribute__ ((__nothrow__));


extern double log10 (double __x) __attribute__ ((__nothrow__)); extern double __log10 (double __x) __attribute__ ((__nothrow__));


extern double modf (double __x, double *__iptr) __attribute__ ((__nothrow__)); extern double __modf (double __x, double *__iptr) __attribute__ ((__nothrow__));

# 127 "/usr/include/bits/mathcalls.h" 3


extern double expm1 (double __x) __attribute__ ((__nothrow__)); extern double __expm1 (double __x) __attribute__ ((__nothrow__));


extern double log1p (double __x) __attribute__ ((__nothrow__)); extern double __log1p (double __x) __attribute__ ((__nothrow__));


extern double logb (double __x) __attribute__ ((__nothrow__)); extern double __logb (double __x) __attribute__ ((__nothrow__));

# 152 "/usr/include/bits/mathcalls.h" 3


extern double pow (double __x, double __y) __attribute__ ((__nothrow__)); extern double __pow (double __x, double __y) __attribute__ ((__nothrow__));


extern double sqrt (double __x) __attribute__ ((__nothrow__)); extern double __sqrt (double __x) __attribute__ ((__nothrow__));





extern double hypot (double __x, double __y) __attribute__ ((__nothrow__)); extern double __hypot (double __x, double __y) __attribute__ ((__nothrow__));






extern double cbrt (double __x) __attribute__ ((__nothrow__)); extern double __cbrt (double __x) __attribute__ ((__nothrow__));








extern double ceil (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern double __ceil (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern double fabs (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern double __fabs (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern double floor (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern double __floor (double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern double fmod (double __x, double __y) __attribute__ ((__nothrow__)); extern double __fmod (double __x, double __y) __attribute__ ((__nothrow__));




extern int __isinf (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int __finite (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern int isinf (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int finite (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern double drem (double __x, double __y) __attribute__ ((__nothrow__)); extern double __drem (double __x, double __y) __attribute__ ((__nothrow__));



extern double significand (double __x) __attribute__ ((__nothrow__)); extern double __significand (double __x) __attribute__ ((__nothrow__));





extern double copysign (double __x, double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern double __copysign (double __x, double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

# 231 "/usr/include/bits/mathcalls.h" 3
extern int __isnan (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));



extern int isnan (double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern double j0 (double) __attribute__ ((__nothrow__)); extern double __j0 (double) __attribute__ ((__nothrow__));
extern double j1 (double) __attribute__ ((__nothrow__)); extern double __j1 (double) __attribute__ ((__nothrow__));
extern double jn (int, double) __attribute__ ((__nothrow__)); extern double __jn (int, double) __attribute__ ((__nothrow__));
extern double y0 (double) __attribute__ ((__nothrow__)); extern double __y0 (double) __attribute__ ((__nothrow__));
extern double y1 (double) __attribute__ ((__nothrow__)); extern double __y1 (double) __attribute__ ((__nothrow__));
extern double yn (int, double) __attribute__ ((__nothrow__)); extern double __yn (int, double) __attribute__ ((__nothrow__));






extern double erf (double) __attribute__ ((__nothrow__)); extern double __erf (double) __attribute__ ((__nothrow__));
extern double erfc (double) __attribute__ ((__nothrow__)); extern double __erfc (double) __attribute__ ((__nothrow__));
extern double lgamma (double) __attribute__ ((__nothrow__)); extern double __lgamma (double) __attribute__ ((__nothrow__));

# 265 "/usr/include/bits/mathcalls.h" 3
extern double gamma (double) __attribute__ ((__nothrow__)); extern double __gamma (double) __attribute__ ((__nothrow__));






extern double lgamma_r (double, int *__signgamp) __attribute__ ((__nothrow__)); extern double __lgamma_r (double, int *__signgamp) __attribute__ ((__nothrow__));







extern double rint (double __x) __attribute__ ((__nothrow__)); extern double __rint (double __x) __attribute__ ((__nothrow__));


extern double nextafter (double __x, double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern double __nextafter (double __x, double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern double remainder (double __x, double __y) __attribute__ ((__nothrow__)); extern double __remainder (double __x, double __y) __attribute__ ((__nothrow__));



extern double scalbn (double __x, int __n) __attribute__ ((__nothrow__)); extern double __scalbn (double __x, int __n) __attribute__ ((__nothrow__));



extern int ilogb (double __x) __attribute__ ((__nothrow__)); extern int __ilogb (double __x) __attribute__ ((__nothrow__));
# 359 "/usr/include/bits/mathcalls.h" 3





extern double scalb (double __x, double __n) __attribute__ ((__nothrow__)); extern double __scalb (double __x, double __n) __attribute__ ((__nothrow__));
# 72 "/usr/include/math.h" 2 3
# 94 "/usr/include/math.h" 3
# 1 "/usr/include/bits/mathcalls.h" 1 3
# 53 "/usr/include/bits/mathcalls.h" 3


extern float acosf (float __x) __attribute__ ((__nothrow__)); extern float __acosf (float __x) __attribute__ ((__nothrow__));

extern float asinf (float __x) __attribute__ ((__nothrow__)); extern float __asinf (float __x) __attribute__ ((__nothrow__));

extern float atanf (float __x) __attribute__ ((__nothrow__)); extern float __atanf (float __x) __attribute__ ((__nothrow__));

extern float atan2f (float __y, float __x) __attribute__ ((__nothrow__)); extern float __atan2f (float __y, float __x) __attribute__ ((__nothrow__));


extern float cosf (float __x) __attribute__ ((__nothrow__)); extern float __cosf (float __x) __attribute__ ((__nothrow__));

extern float sinf (float __x) __attribute__ ((__nothrow__)); extern float __sinf (float __x) __attribute__ ((__nothrow__));

extern float tanf (float __x) __attribute__ ((__nothrow__)); extern float __tanf (float __x) __attribute__ ((__nothrow__));




extern float coshf (float __x) __attribute__ ((__nothrow__)); extern float __coshf (float __x) __attribute__ ((__nothrow__));

extern float sinhf (float __x) __attribute__ ((__nothrow__)); extern float __sinhf (float __x) __attribute__ ((__nothrow__));

extern float tanhf (float __x) __attribute__ ((__nothrow__)); extern float __tanhf (float __x) __attribute__ ((__nothrow__));

# 87 "/usr/include/bits/mathcalls.h" 3


extern float acoshf (float __x) __attribute__ ((__nothrow__)); extern float __acoshf (float __x) __attribute__ ((__nothrow__));

extern float asinhf (float __x) __attribute__ ((__nothrow__)); extern float __asinhf (float __x) __attribute__ ((__nothrow__));

extern float atanhf (float __x) __attribute__ ((__nothrow__)); extern float __atanhf (float __x) __attribute__ ((__nothrow__));







extern float expf (float __x) __attribute__ ((__nothrow__)); extern float __expf (float __x) __attribute__ ((__nothrow__));


extern float frexpf (float __x, int *__exponent) __attribute__ ((__nothrow__)); extern float __frexpf (float __x, int *__exponent) __attribute__ ((__nothrow__));


extern float ldexpf (float __x, int __exponent) __attribute__ ((__nothrow__)); extern float __ldexpf (float __x, int __exponent) __attribute__ ((__nothrow__));


extern float logf (float __x) __attribute__ ((__nothrow__)); extern float __logf (float __x) __attribute__ ((__nothrow__));


extern float log10f (float __x) __attribute__ ((__nothrow__)); extern float __log10f (float __x) __attribute__ ((__nothrow__));


extern float modff (float __x, float *__iptr) __attribute__ ((__nothrow__)); extern float __modff (float __x, float *__iptr) __attribute__ ((__nothrow__));

# 127 "/usr/include/bits/mathcalls.h" 3


extern float expm1f (float __x) __attribute__ ((__nothrow__)); extern float __expm1f (float __x) __attribute__ ((__nothrow__));


extern float log1pf (float __x) __attribute__ ((__nothrow__)); extern float __log1pf (float __x) __attribute__ ((__nothrow__));


extern float logbf (float __x) __attribute__ ((__nothrow__)); extern float __logbf (float __x) __attribute__ ((__nothrow__));

# 152 "/usr/include/bits/mathcalls.h" 3


extern float powf (float __x, float __y) __attribute__ ((__nothrow__)); extern float __powf (float __x, float __y) __attribute__ ((__nothrow__));


extern float sqrtf (float __x) __attribute__ ((__nothrow__)); extern float __sqrtf (float __x) __attribute__ ((__nothrow__));





extern float hypotf (float __x, float __y) __attribute__ ((__nothrow__)); extern float __hypotf (float __x, float __y) __attribute__ ((__nothrow__));






extern float cbrtf (float __x) __attribute__ ((__nothrow__)); extern float __cbrtf (float __x) __attribute__ ((__nothrow__));








extern float ceilf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern float __ceilf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern float fabsf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern float __fabsf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern float floorf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern float __floorf (float __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern float fmodf (float __x, float __y) __attribute__ ((__nothrow__)); extern float __fmodf (float __x, float __y) __attribute__ ((__nothrow__));




extern int __isinff (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int __finitef (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern int isinff (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int finitef (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern float dremf (float __x, float __y) __attribute__ ((__nothrow__)); extern float __dremf (float __x, float __y) __attribute__ ((__nothrow__));



extern float significandf (float __x) __attribute__ ((__nothrow__)); extern float __significandf (float __x) __attribute__ ((__nothrow__));





extern float copysignf (float __x, float __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern float __copysignf (float __x, float __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

# 231 "/usr/include/bits/mathcalls.h" 3
extern int __isnanf (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));



extern int isnanf (float __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern float j0f (float) __attribute__ ((__nothrow__)); extern float __j0f (float) __attribute__ ((__nothrow__));
extern float j1f (float) __attribute__ ((__nothrow__)); extern float __j1f (float) __attribute__ ((__nothrow__));
extern float jnf (int, float) __attribute__ ((__nothrow__)); extern float __jnf (int, float) __attribute__ ((__nothrow__));
extern float y0f (float) __attribute__ ((__nothrow__)); extern float __y0f (float) __attribute__ ((__nothrow__));
extern float y1f (float) __attribute__ ((__nothrow__)); extern float __y1f (float) __attribute__ ((__nothrow__));
extern float ynf (int, float) __attribute__ ((__nothrow__)); extern float __ynf (int, float) __attribute__ ((__nothrow__));






extern float erff (float) __attribute__ ((__nothrow__)); extern float __erff (float) __attribute__ ((__nothrow__));
extern float erfcf (float) __attribute__ ((__nothrow__)); extern float __erfcf (float) __attribute__ ((__nothrow__));
extern float lgammaf (float) __attribute__ ((__nothrow__)); extern float __lgammaf (float) __attribute__ ((__nothrow__));

# 265 "/usr/include/bits/mathcalls.h" 3
extern float gammaf (float) __attribute__ ((__nothrow__)); extern float __gammaf (float) __attribute__ ((__nothrow__));






extern float lgammaf_r (float, int *__signgamp) __attribute__ ((__nothrow__)); extern float __lgammaf_r (float, int *__signgamp) __attribute__ ((__nothrow__));







extern float rintf (float __x) __attribute__ ((__nothrow__)); extern float __rintf (float __x) __attribute__ ((__nothrow__));


extern float nextafterf (float __x, float __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern float __nextafterf (float __x, float __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern float remainderf (float __x, float __y) __attribute__ ((__nothrow__)); extern float __remainderf (float __x, float __y) __attribute__ ((__nothrow__));



extern float scalbnf (float __x, int __n) __attribute__ ((__nothrow__)); extern float __scalbnf (float __x, int __n) __attribute__ ((__nothrow__));



extern int ilogbf (float __x) __attribute__ ((__nothrow__)); extern int __ilogbf (float __x) __attribute__ ((__nothrow__));
# 359 "/usr/include/bits/mathcalls.h" 3





extern float scalbf (float __x, float __n) __attribute__ ((__nothrow__)); extern float __scalbf (float __x, float __n) __attribute__ ((__nothrow__));
# 95 "/usr/include/math.h" 2 3
# 141 "/usr/include/math.h" 3
# 1 "/usr/include/bits/mathcalls.h" 1 3
# 53 "/usr/include/bits/mathcalls.h" 3


extern long double acosl (long double __x) __attribute__ ((__nothrow__)); extern long double __acosl (long double __x) __attribute__ ((__nothrow__));

extern long double asinl (long double __x) __attribute__ ((__nothrow__)); extern long double __asinl (long double __x) __attribute__ ((__nothrow__));

extern long double atanl (long double __x) __attribute__ ((__nothrow__)); extern long double __atanl (long double __x) __attribute__ ((__nothrow__));

extern long double atan2l (long double __y, long double __x) __attribute__ ((__nothrow__)); extern long double __atan2l (long double __y, long double __x) __attribute__ ((__nothrow__));


extern long double cosl (long double __x) __attribute__ ((__nothrow__)); extern long double __cosl (long double __x) __attribute__ ((__nothrow__));

extern long double sinl (long double __x) __attribute__ ((__nothrow__)); extern long double __sinl (long double __x) __attribute__ ((__nothrow__));

extern long double tanl (long double __x) __attribute__ ((__nothrow__)); extern long double __tanl (long double __x) __attribute__ ((__nothrow__));




extern long double coshl (long double __x) __attribute__ ((__nothrow__)); extern long double __coshl (long double __x) __attribute__ ((__nothrow__));

extern long double sinhl (long double __x) __attribute__ ((__nothrow__)); extern long double __sinhl (long double __x) __attribute__ ((__nothrow__));

extern long double tanhl (long double __x) __attribute__ ((__nothrow__)); extern long double __tanhl (long double __x) __attribute__ ((__nothrow__));

# 87 "/usr/include/bits/mathcalls.h" 3


extern long double acoshl (long double __x) __attribute__ ((__nothrow__)); extern long double __acoshl (long double __x) __attribute__ ((__nothrow__));

extern long double asinhl (long double __x) __attribute__ ((__nothrow__)); extern long double __asinhl (long double __x) __attribute__ ((__nothrow__));

extern long double atanhl (long double __x) __attribute__ ((__nothrow__)); extern long double __atanhl (long double __x) __attribute__ ((__nothrow__));







extern long double expl (long double __x) __attribute__ ((__nothrow__)); extern long double __expl (long double __x) __attribute__ ((__nothrow__));


extern long double frexpl (long double __x, int *__exponent) __attribute__ ((__nothrow__)); extern long double __frexpl (long double __x, int *__exponent) __attribute__ ((__nothrow__));


extern long double ldexpl (long double __x, int __exponent) __attribute__ ((__nothrow__)); extern long double __ldexpl (long double __x, int __exponent) __attribute__ ((__nothrow__));


extern long double logl (long double __x) __attribute__ ((__nothrow__)); extern long double __logl (long double __x) __attribute__ ((__nothrow__));


extern long double log10l (long double __x) __attribute__ ((__nothrow__)); extern long double __log10l (long double __x) __attribute__ ((__nothrow__));


extern long double modfl (long double __x, long double *__iptr) __attribute__ ((__nothrow__)); extern long double __modfl (long double __x, long double *__iptr) __attribute__ ((__nothrow__));

# 127 "/usr/include/bits/mathcalls.h" 3


extern long double expm1l (long double __x) __attribute__ ((__nothrow__)); extern long double __expm1l (long double __x) __attribute__ ((__nothrow__));


extern long double log1pl (long double __x) __attribute__ ((__nothrow__)); extern long double __log1pl (long double __x) __attribute__ ((__nothrow__));


extern long double logbl (long double __x) __attribute__ ((__nothrow__)); extern long double __logbl (long double __x) __attribute__ ((__nothrow__));

# 152 "/usr/include/bits/mathcalls.h" 3


extern long double powl (long double __x, long double __y) __attribute__ ((__nothrow__)); extern long double __powl (long double __x, long double __y) __attribute__ ((__nothrow__));


extern long double sqrtl (long double __x) __attribute__ ((__nothrow__)); extern long double __sqrtl (long double __x) __attribute__ ((__nothrow__));





extern long double hypotl (long double __x, long double __y) __attribute__ ((__nothrow__)); extern long double __hypotl (long double __x, long double __y) __attribute__ ((__nothrow__));






extern long double cbrtl (long double __x) __attribute__ ((__nothrow__)); extern long double __cbrtl (long double __x) __attribute__ ((__nothrow__));








extern long double ceill (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern long double __ceill (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern long double fabsl (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern long double __fabsl (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern long double floorl (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern long double __floorl (long double __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern long double fmodl (long double __x, long double __y) __attribute__ ((__nothrow__)); extern long double __fmodl (long double __x, long double __y) __attribute__ ((__nothrow__));




extern int __isinfl (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int __finitel (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern int isinfl (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern int finitel (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern long double dreml (long double __x, long double __y) __attribute__ ((__nothrow__)); extern long double __dreml (long double __x, long double __y) __attribute__ ((__nothrow__));



extern long double significandl (long double __x) __attribute__ ((__nothrow__)); extern long double __significandl (long double __x) __attribute__ ((__nothrow__));





extern long double copysignl (long double __x, long double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern long double __copysignl (long double __x, long double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

# 231 "/usr/include/bits/mathcalls.h" 3
extern int __isnanl (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));



extern int isnanl (long double __value) __attribute__ ((__nothrow__)) __attribute__ ((__const__));


extern long double j0l (long double) __attribute__ ((__nothrow__)); extern long double __j0l (long double) __attribute__ ((__nothrow__));
extern long double j1l (long double) __attribute__ ((__nothrow__)); extern long double __j1l (long double) __attribute__ ((__nothrow__));
extern long double jnl (int, long double) __attribute__ ((__nothrow__)); extern long double __jnl (int, long double) __attribute__ ((__nothrow__));
extern long double y0l (long double) __attribute__ ((__nothrow__)); extern long double __y0l (long double) __attribute__ ((__nothrow__));
extern long double y1l (long double) __attribute__ ((__nothrow__)); extern long double __y1l (long double) __attribute__ ((__nothrow__));
extern long double ynl (int, long double) __attribute__ ((__nothrow__)); extern long double __ynl (int, long double) __attribute__ ((__nothrow__));






extern long double erfl (long double) __attribute__ ((__nothrow__)); extern long double __erfl (long double) __attribute__ ((__nothrow__));
extern long double erfcl (long double) __attribute__ ((__nothrow__)); extern long double __erfcl (long double) __attribute__ ((__nothrow__));
extern long double lgammal (long double) __attribute__ ((__nothrow__)); extern long double __lgammal (long double) __attribute__ ((__nothrow__));

# 265 "/usr/include/bits/mathcalls.h" 3
extern long double gammal (long double) __attribute__ ((__nothrow__)); extern long double __gammal (long double) __attribute__ ((__nothrow__));






extern long double lgammal_r (long double, int *__signgamp) __attribute__ ((__nothrow__)); extern long double __lgammal_r (long double, int *__signgamp) __attribute__ ((__nothrow__));







extern long double rintl (long double __x) __attribute__ ((__nothrow__)); extern long double __rintl (long double __x) __attribute__ ((__nothrow__));


extern long double nextafterl (long double __x, long double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__)); extern long double __nextafterl (long double __x, long double __y) __attribute__ ((__nothrow__)) __attribute__ ((__const__));





extern long double remainderl (long double __x, long double __y) __attribute__ ((__nothrow__)); extern long double __remainderl (long double __x, long double __y) __attribute__ ((__nothrow__));



extern long double scalbnl (long double __x, int __n) __attribute__ ((__nothrow__)); extern long double __scalbnl (long double __x, int __n) __attribute__ ((__nothrow__));



extern int ilogbl (long double __x) __attribute__ ((__nothrow__)); extern int __ilogbl (long double __x) __attribute__ ((__nothrow__));
# 359 "/usr/include/bits/mathcalls.h" 3





extern long double scalbl (long double __x, long double __n) __attribute__ ((__nothrow__)); extern long double __scalbl (long double __x, long double __n) __attribute__ ((__nothrow__));
# 142 "/usr/include/math.h" 2 3
# 157 "/usr/include/math.h" 3
extern int signgam;
# 284 "/usr/include/math.h" 3
typedef enum
{
  _IEEE_ = -1,
  _SVID_,
  _XOPEN_,
  _POSIX_,
  _ISOC_
} _LIB_VERSION_TYPE;




extern _LIB_VERSION_TYPE _LIB_VERSION;
# 309 "/usr/include/math.h" 3
struct exception

  {
    int type;
    char *name;
    double arg1;
    double arg2;
    double retval;
  };




extern int matherr (struct exception *__exc);
# 465 "/usr/include/math.h" 3

# 11 "swarm_isort64.comb.c" 2
# 1 "/usr/include/errno.h" 1 3
# 12 "swarm_isort64.comb.c" 2
# 1 "/usr/include/sys/time.h" 1 3
# 29 "/usr/include/sys/time.h" 3
# 1 "/usr/include/bits/time.h" 1 3
# 30 "/usr/include/sys/time.h" 2 3
# 39 "/usr/include/sys/time.h" 3

# 57 "/usr/include/sys/time.h" 3
struct timezone
  {
    int tz_minuteswest;
    int tz_dsttime;
  };

typedef struct timezone *__restrict __timezone_ptr_t;
# 73 "/usr/include/sys/time.h" 3
extern int gettimeofday (struct timeval *__restrict __tv,
    __timezone_ptr_t __tz) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int settimeofday (__const struct timeval *__tv,
    __const struct timezone *__tz)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int adjtime (__const struct timeval *__delta,
      struct timeval *__olddelta) __attribute__ ((__nothrow__));




enum __itimer_which
  {

    ITIMER_REAL = 0,


    ITIMER_VIRTUAL = 1,



    ITIMER_PROF = 2

  };



struct itimerval
  {

    struct timeval it_interval;

    struct timeval it_value;
  };






typedef int __itimer_which_t;




extern int getitimer (__itimer_which_t __which,
        struct itimerval *__value) __attribute__ ((__nothrow__));




extern int setitimer (__itimer_which_t __which,
        __const struct itimerval *__restrict __new,
        struct itimerval *__restrict __old) __attribute__ ((__nothrow__));




extern int utimes (__const char *__file, __const struct timeval __tvp[2])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int lutimes (__const char *__file, __const struct timeval __tvp[2])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int futimes (int __fd, __const struct timeval __tvp[2]) __attribute__ ((__nothrow__));
# 191 "/usr/include/sys/time.h" 3

# 13 "swarm_isort64.comb.c" 2
# 1 "/usr/include/unistd.h" 1 3
# 28 "/usr/include/unistd.h" 3

# 173 "/usr/include/unistd.h" 3
# 1 "/usr/include/bits/posix_opt.h" 1 3
# 174 "/usr/include/unistd.h" 2 3
# 197 "/usr/include/unistd.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 198 "/usr/include/unistd.h" 2 3
# 226 "/usr/include/unistd.h" 3
typedef __useconds_t useconds_t;
# 238 "/usr/include/unistd.h" 3
typedef __intptr_t intptr_t;






typedef __socklen_t socklen_t;
# 258 "/usr/include/unistd.h" 3
extern int access (__const char *__name, int __type) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 301 "/usr/include/unistd.h" 3
extern __off_t lseek (int __fd, __off_t __offset, int __whence) __attribute__ ((__nothrow__));
# 320 "/usr/include/unistd.h" 3
extern int close (int __fd);






extern ssize_t read (int __fd, void *__buf, size_t __nbytes) ;





extern ssize_t write (int __fd, __const void *__buf, size_t __n) ;
# 384 "/usr/include/unistd.h" 3
extern int pipe (int __pipedes[2]) __attribute__ ((__nothrow__)) ;
# 399 "/usr/include/unistd.h" 3
extern unsigned int alarm (unsigned int __seconds) __attribute__ ((__nothrow__));
# 411 "/usr/include/unistd.h" 3
extern unsigned int sleep (unsigned int __seconds);






extern __useconds_t ualarm (__useconds_t __value, __useconds_t __interval)
     __attribute__ ((__nothrow__));






extern int usleep (__useconds_t __useconds);
# 435 "/usr/include/unistd.h" 3
extern int pause (void);



extern int chown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchown (int __fd, __uid_t __owner, __gid_t __group) __attribute__ ((__nothrow__)) ;




extern int lchown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 463 "/usr/include/unistd.h" 3
extern int chdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchdir (int __fd) __attribute__ ((__nothrow__)) ;
# 477 "/usr/include/unistd.h" 3
extern char *getcwd (char *__buf, size_t __size) __attribute__ ((__nothrow__)) ;
# 490 "/usr/include/unistd.h" 3
extern char *getwd (char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__)) ;




extern int dup (int __fd) __attribute__ ((__nothrow__)) ;


extern int dup2 (int __fd, int __fd2) __attribute__ ((__nothrow__));
# 508 "/usr/include/unistd.h" 3
extern char **__environ;







extern int execve (__const char *__path, char *__const __argv[],
     char *__const __envp[]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 528 "/usr/include/unistd.h" 3
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





# 1 "/usr/include/bits/confname.h" 1 3
# 26 "/usr/include/bits/confname.h" 3
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
# 567 "/usr/include/unistd.h" 2 3


extern long int pathconf (__const char *__path, int __name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int fpathconf (int __fd, int __name) __attribute__ ((__nothrow__));


extern long int sysconf (int __name) __attribute__ ((__nothrow__));



extern size_t confstr (int __name, char *__buf, size_t __len) __attribute__ ((__nothrow__));




extern __pid_t getpid (void) __attribute__ ((__nothrow__));


extern __pid_t getppid (void) __attribute__ ((__nothrow__));




extern __pid_t getpgrp (void) __attribute__ ((__nothrow__));
# 603 "/usr/include/unistd.h" 3
extern __pid_t __getpgid (__pid_t __pid) __attribute__ ((__nothrow__));
# 612 "/usr/include/unistd.h" 3
extern int setpgid (__pid_t __pid, __pid_t __pgid) __attribute__ ((__nothrow__));
# 629 "/usr/include/unistd.h" 3
extern int setpgrp (void) __attribute__ ((__nothrow__));
# 646 "/usr/include/unistd.h" 3
extern __pid_t setsid (void) __attribute__ ((__nothrow__));







extern __uid_t getuid (void) __attribute__ ((__nothrow__));


extern __uid_t geteuid (void) __attribute__ ((__nothrow__));


extern __gid_t getgid (void) __attribute__ ((__nothrow__));


extern __gid_t getegid (void) __attribute__ ((__nothrow__));




extern int getgroups (int __size, __gid_t __list[]) __attribute__ ((__nothrow__)) ;
# 679 "/usr/include/unistd.h" 3
extern int setuid (__uid_t __uid) __attribute__ ((__nothrow__));




extern int setreuid (__uid_t __ruid, __uid_t __euid) __attribute__ ((__nothrow__));




extern int seteuid (__uid_t __uid) __attribute__ ((__nothrow__));






extern int setgid (__gid_t __gid) __attribute__ ((__nothrow__));




extern int setregid (__gid_t __rgid, __gid_t __egid) __attribute__ ((__nothrow__));




extern int setegid (__gid_t __gid) __attribute__ ((__nothrow__));
# 735 "/usr/include/unistd.h" 3
extern __pid_t fork (void) __attribute__ ((__nothrow__));






extern __pid_t vfork (void) __attribute__ ((__nothrow__));





extern char *ttyname (int __fd) __attribute__ ((__nothrow__));



extern int ttyname_r (int __fd, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;



extern int isatty (int __fd) __attribute__ ((__nothrow__));





extern int ttyslot (void) __attribute__ ((__nothrow__));




extern int link (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
# 781 "/usr/include/unistd.h" 3
extern int symlink (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern ssize_t readlink (__const char *__restrict __path,
    char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;
# 804 "/usr/include/unistd.h" 3
extern int unlink (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 813 "/usr/include/unistd.h" 3
extern int rmdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern __pid_t tcgetpgrp (int __fd) __attribute__ ((__nothrow__));


extern int tcsetpgrp (int __fd, __pid_t __pgrp_id) __attribute__ ((__nothrow__));






extern char *getlogin (void);







extern int getlogin_r (char *__name, size_t __name_len) __attribute__ ((__nonnull__ (1)));




extern int setlogin (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 849 "/usr/include/unistd.h" 3
# 1 "./getopt.h" 1 3
# 36 "./getopt.h" 3
extern char *optarg;
# 50 "./getopt.h" 3
extern int optind;




extern int opterr;



extern int optopt;
# 134 "./getopt.h" 3
extern int getopt (int __argc, char *const *__argv, const char *__shortopts);
# 850 "/usr/include/unistd.h" 2 3







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
# 935 "/usr/include/unistd.h" 3
extern int fsync (int __fd);






extern long int gethostid (void);


extern void sync (void) __attribute__ ((__nothrow__));




extern int getpagesize (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int getdtablesize (void) __attribute__ ((__nothrow__));




extern int truncate (__const char *__file, __off_t __length)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 982 "/usr/include/unistd.h" 3
extern int ftruncate (int __fd, __off_t __length) __attribute__ ((__nothrow__)) ;
# 1002 "/usr/include/unistd.h" 3
extern int brk (void *__addr) __attribute__ ((__nothrow__)) ;





extern void *sbrk (intptr_t __delta) __attribute__ ((__nothrow__));
# 1023 "/usr/include/unistd.h" 3
extern long int syscall (long int __sysno, ...) __attribute__ ((__nothrow__));
# 1046 "/usr/include/unistd.h" 3
extern int lockf (int __fd, int __cmd, __off_t __len) ;
# 1077 "/usr/include/unistd.h" 3
extern int fdatasync (int __fildes);
# 1115 "/usr/include/unistd.h" 3

# 14 "swarm_isort64.comb.c" 2
# 1 "./getopt.h" 1 3
# 36 "./getopt.h" 3
extern char *optarg;
# 50 "./getopt.h" 3
extern int optind;




extern int opterr;



extern int optopt;
# 83 "./getopt.h" 3
struct option
{

  const char *name;





  int has_arg;
  int *flag;
  int val;
};
# 134 "./getopt.h" 3
extern int getopt (int __argc, char *const *__argv, const char *__shortopts);





extern int getopt_long (int __argc, char *const *__argv, const char *__shortopts,
          const struct option *__longopts, int *__longind);
extern int getopt_long_only (int __argc, char *const *__argv,
        const char *__shortopts,
               const struct option *__longopts, int *__longind);


extern int _getopt_internal (int __argc, char *const *__argv,
        const char *__shortopts,
               const struct option *__longopts, int *__longind,
        int __long_only);
# 15 "swarm_isort64.comb.c" 2
# 65 "swarm_isort64.comb.c"
enum reduce_tag {MAX, MIN, SUM, PROD, LAND, BAND, LOR, BOR, LXOR, BXOR};

extern FILE* SWARM_outfile;
# 103 "swarm_isort64.comb.c"
typedef struct {
  long *randtbl;
  long *fptr;
  long *rptr;
  long *state;
  int rand_type;
  int rand_deg;
  int rand_sep;
  long *end_ptr;
} rrandom_info_t;





extern int MAXTHREADS;

extern int THREADS;

struct thread_inf {
  int mythread;
  int argc;
  char **argv;
  long m1;
  long m2;
  long blk;






  rrandom_info_t rand;


  long rbs;
  short rc;
  int udata;
};

typedef struct thread_inf uthread_info_t;

typedef int reduce_t;

extern uthread_info_t *uthread_info;
# 187 "swarm_isort64.comb.c"
void SWARM_Barrier_tree(uthread_info_t *ti);
void SWARM_Barrier_sync(uthread_info_t *ti);
void *SWARM_malloc(int bytes, uthread_info_t *ti);
void *SWARM_malloc_l(long bytes, uthread_info_t *ti);
void SWARM_free(void *, uthread_info_t *ti);

typedef pthread_mutex_t SWARM_mutex_t;
typedef pthread_mutexattr_t SWARM_mutexattr_t;
int SWARM_mutex_init(SWARM_mutex_t **, const SWARM_mutexattr_t *, uthread_info_t *ti);
int SWARM_mutex_destroy(SWARM_mutex_t *, uthread_info_t *ti);




int SWARM_Bcast_i(int myval, uthread_info_t *ti);
long SWARM_Bcast_l(long myval, uthread_info_t *ti);
double SWARM_Bcast_d(double myval, uthread_info_t *ti);
char SWARM_Bcast_c(char myval, uthread_info_t *ti);
int *SWARM_Bcast_ip(int *myval, uthread_info_t *ti);
long *SWARM_Bcast_lp(long *myval, uthread_info_t *ti);
double *SWARM_Bcast_dp(double *myval, uthread_info_t *ti);
char *SWARM_Bcast_cp(char *myval, uthread_info_t *ti);
int SWARM_Reduce_i(int myval, reduce_t op, uthread_info_t *ti);
long SWARM_Reduce_l(long myval, reduce_t op, uthread_info_t *ti);
double SWARM_Reduce_d(double myval, reduce_t op, uthread_info_t *ti);
int SWARM_Scan_i(int myval, reduce_t op, uthread_info_t *ti);
long SWARM_Scan_l(long myval, reduce_t op, uthread_info_t *ti);
double SWARM_Scan_d(double myval, reduce_t op, uthread_info_t *ti);

void SWARM_Init(int*, char***);
void SWARM_Run(void *);
void SWARM_Finalize(void);
void SWARM_Cleanup(uthread_info_t *ti);

void assert_malloc(void *ptr);
double get_seconds(void);
# 244 "swarm_isort64.comb.c"
typedef struct _SWARM_MULTICORE_barrier {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_barrier_t;

_SWARM_MULTICORE_barrier_t _SWARM_MULTICORE_barrier_init(int n_clients);
void _SWARM_MULTICORE_barrier_destroy(_SWARM_MULTICORE_barrier_t nbarrier);
void _SWARM_MULTICORE_barrier_wait(_SWARM_MULTICORE_barrier_t nbarrier);

typedef struct _SWARM_MULTICORE_reduce_i_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  int sum;
  int result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_reduce_i_t;

_SWARM_MULTICORE_reduce_i_t _SWARM_MULTICORE_reduce_init_i(int n_clients);
void _SWARM_MULTICORE_reduce_destroy_i(_SWARM_MULTICORE_reduce_i_t nbarrier);
int _SWARM_MULTICORE_reduce_i(_SWARM_MULTICORE_reduce_i_t nbarrier, int val, reduce_t op);

typedef struct _SWARM_MULTICORE_reduce_l_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  long sum;
  long result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_reduce_l_t;

_SWARM_MULTICORE_reduce_l_t _SWARM_MULTICORE_reduce_init_l(int n_clients);
void _SWARM_MULTICORE_reduce_destroy_l(_SWARM_MULTICORE_reduce_l_t nbarrier);
long _SWARM_MULTICORE_reduce_l(_SWARM_MULTICORE_reduce_l_t nbarrier, long val, reduce_t op);

typedef struct _SWARM_MULTICORE_reduce_d_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  double sum;
  double result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_reduce_d_t;

_SWARM_MULTICORE_reduce_d_t _SWARM_MULTICORE_reduce_init_d(int n_clients);
void _SWARM_MULTICORE_reduce_destroy_d(_SWARM_MULTICORE_reduce_d_t nbarrier);
double _SWARM_MULTICORE_reduce_d(_SWARM_MULTICORE_reduce_d_t nbarrier, double val, reduce_t op);

typedef struct _SWARM_MULTICORE_scan_i_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  int *result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_scan_i_t;

_SWARM_MULTICORE_scan_i_t _SWARM_MULTICORE_scan_init_i(int n_clients);
void _SWARM_MULTICORE_scan_destroy_i(_SWARM_MULTICORE_scan_i_t nbarrier);
int _SWARM_MULTICORE_scan_i(_SWARM_MULTICORE_scan_i_t nbarrier, int val, reduce_t op,int th_index);


typedef struct _SWARM_MULTICORE_scan_l_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  long *result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_scan_l_t;

_SWARM_MULTICORE_scan_l_t _SWARM_MULTICORE_scan_init_l(int n_clients);
void _SWARM_MULTICORE_scan_destroy_l(_SWARM_MULTICORE_scan_l_t nbarrier);
long _SWARM_MULTICORE_scan_l(_SWARM_MULTICORE_scan_l_t nbarrier, long val, reduce_t op,int th_index);

typedef struct _SWARM_MULTICORE_scan_d_s {
  pthread_mutex_t lock;
  int n_clients;
  int n_waiting;
  int phase;
  double *result;
  pthread_cond_t wait_cv;
} *_SWARM_MULTICORE_scan_d_t;

_SWARM_MULTICORE_scan_d_t _SWARM_MULTICORE_scan_init_d(int n_clients);
void _SWARM_MULTICORE_scan_destroy_d(_SWARM_MULTICORE_scan_d_t nbarrier);
double _SWARM_MULTICORE_scan_d(_SWARM_MULTICORE_scan_d_t nbarrier, double val, reduce_t op,int th_index);

typedef struct _SWARM_MULTICORE_spin_barrier {
  int n_clients;
  pthread_mutex_t lock;
  int n_waiting;
  int phase;
} *_SWARM_MULTICORE_spin_barrier_t;

_SWARM_MULTICORE_spin_barrier_t _SWARM_MULTICORE_spin_barrier_init(int n_clients);
void _SWARM_MULTICORE_spin_barrier_destroy(_SWARM_MULTICORE_spin_barrier_t sbarrier);
void _SWARM_MULTICORE_spin_barrier_wait(_SWARM_MULTICORE_spin_barrier_t sbarrier);




void countsort_swarm(long q,
       int *lKey,
       int *lSorted,
       int R,
       int bitOff, int m,
       uthread_info_t *ti);



void radixsort_swarm_s3(long q,
   int *lKeys,
   int *lSorted,
   uthread_info_t *ti);
void radixsort_swarm_s2(long q,
   int *lKeys,
   int *lSorted,
   uthread_info_t *ti);
void radixsort20_swarm_s1(long q,
     int *lKeys,
     int *lSorted,
     uthread_info_t *ti);
void radixsort20_swarm_s2(long q,
     int *lKeys,
     int *lSorted,
     uthread_info_t *ti);

void radixsort_check(long q,
       int *lSorted);
# 391 "swarm_isort64.comb.c"
void countsort_swarm(long q,
       int *lKey,
       int *lSorted,
       int R,
       int bitOff, int m,
       uthread_info_t *ti)



{
    register int
 j,
 k,
        last, temp,
 offset;

    int *myHisto,
        *mhp,
        *mps,
        *psHisto,
        *allHisto;

    long x;

    myHisto = (int *)SWARM_malloc(THREADS*R*sizeof(int), ti);
    psHisto = (int *)SWARM_malloc(THREADS*R*sizeof(int), ti);

    mhp = myHisto + (ti->mythread)*R;

    for (k=0 ; k<R ; k++)
      mhp[k] = 0;

    if ((((0))==0)&&(((q))==THREADS)) { ti->m1 = (ti->mythread); ti->m2 = ti->m1 + 1; } else { ti->blk = (((q))-((0)))/THREADS; if (ti->blk == 0) { ti->m1 = ((0))+(ti->mythread); ti->m2 = (ti->m1) + 1; if ((ti->m1) >= ((q))) ti->m1 = ti->m2; } else { ti->m1 = (ti->blk) * (ti->mythread) + ((0)); if ((ti->mythread) < THREADS-1) ti->m2 = (ti->m1)+(ti->blk); else ti->m2 = ((q)); } } if (((1))>1) { while ((ti->m1-((0))) % ((1)) > 0) ti->m1 += 1; } for ((x)=ti->m1 ; (x)<ti->m2 ; (x)+=((1)))
      mhp[((lKey[x]>>bitOff) & ~(~0<<m))]++;

    SWARM_Barrier_sync(ti);

    if ((((0))==0)&&(((R))==THREADS)) { ti->m1 = (ti->mythread); ti->m2 = ti->m1 + 1; } else { ti->blk = (((R))-((0)))/THREADS; if (ti->blk == 0) { ti->m1 = ((0))+(ti->mythread); ti->m2 = (ti->m1) + 1; if ((ti->m1) >= ((R))) ti->m1 = ti->m2; } else { ti->m1 = (ti->blk) * (ti->mythread) + ((0)); if ((ti->mythread) < THREADS-1) ti->m2 = (ti->m1)+(ti->blk); else ti->m2 = ((R)); } } if (((1))>1) { while ((ti->m1-((0))) % ((1)) > 0) ti->m1 += 1; } for ((k)=ti->m1 ; (k)<ti->m2 ; (k)+=((1))) {
      last = psHisto[k] = myHisto[k];
      for (j=1 ; j<THREADS ; j++) {
 temp = psHisto[j*R + k] = last + myHisto[j*R + k];
 last = temp;
      }
    }

    allHisto = psHisto+(THREADS-1)*R;

    SWARM_Barrier_sync(ti);

    offset = 0;

    mps = psHisto + ((ti->mythread)*R);
    for (k=0 ; k<R ; k++) {
      mhp[k] = (mps[k] - mhp[k]) + offset;
      offset += allHisto[k];
    }

    SWARM_Barrier_sync(ti);

    if ((((0))==0)&&(((q))==THREADS)) { ti->m1 = (ti->mythread); ti->m2 = ti->m1 + 1; } else { ti->blk = (((q))-((0)))/THREADS; if (ti->blk == 0) { ti->m1 = ((0))+(ti->mythread); ti->m2 = (ti->m1) + 1; if ((ti->m1) >= ((q))) ti->m1 = ti->m2; } else { ti->m1 = (ti->blk) * (ti->mythread) + ((0)); if ((ti->mythread) < THREADS-1) ti->m2 = (ti->m1)+(ti->blk); else ti->m2 = ((q)); } } if (((1))>1) { while ((ti->m1-((0))) % ((1)) > 0) ti->m1 += 1; } for ((x)=ti->m1 ; (x)<ti->m2 ; (x)+=((1))) {
      j = ((lKey[x]>>bitOff) & ~(~0<<m));
      lSorted[mhp[j]] = lKey[x];
      mhp[j]++;
    }

    SWARM_Barrier_sync(ti);

    SWARM_free(psHisto, ti);
    SWARM_free(myHisto, ti);
}


void radixsort_check(long q,
       int *lSorted)

{
  long i;

  for (i=1; i<q ; i++)
    if (lSorted[i-1] > lSorted[i]) {
      fprintf(stderr,
       "ERROR: q:%ld lSorted[%6ld] > lSorted[%6ld] (%6d,%6d)\n",
       q,i-1,i,lSorted[i-1],lSorted[i]);
    }
}


void radixsort_swarm_s3(long q,
   int *lKeys,
   int *lSorted,
   uthread_info_t *ti)

{
  int *lTemp;

  lTemp = (int *)SWARM_malloc_l(q*sizeof(int), ti);

  countsort_swarm(q, lKeys, lSorted, (1<<11), 0, 11, ti);
  countsort_swarm(q, lSorted, lTemp, (1<<11), 11, 11, ti);
  countsort_swarm(q, lTemp, lSorted, (1<<10), 22, 10, ti);

  SWARM_free(lTemp, ti);
}


void radixsort_swarm_s2(long q,
   int *lKeys,
   int *lSorted,
   uthread_info_t *ti)

{
  int *lTemp;

  lTemp = (int *)SWARM_malloc_l(q*sizeof(int), ti);

  countsort_swarm(q, lKeys, lTemp, (1<<16), 0, 16, ti);
  countsort_swarm(q, lTemp, lSorted, (1<<16), 16, 16, ti);

  SWARM_free(lTemp, ti);
}


void radixsort20_swarm_s1(long q,
     int *lKeys,
     int *lSorted,
     uthread_info_t *ti)

{
  countsort_swarm(q, lKeys, lSorted, (1<<20), 0, 20, ti);
}


void radixsort20_swarm_s2(long q,
        int *lKeys,
        int *lSorted,
        uthread_info_t *ti)

{
  int *lTemp;

  lTemp = (int *)SWARM_malloc_l(q*sizeof(int), ti);

  countsort_swarm(q, lKeys, lTemp, (1<<10), 0, 10, ti);
  countsort_swarm(q, lTemp, lSorted, (1<<10), 10, 10, ti);

  SWARM_free(lTemp, ti);
}
# 547 "swarm_isort64.comb.c"
double find_my_seed( long kn,
                       long np,
                       long nn,
                       double s,
                       double a );

void create_seq( double seed, double a , int q, int *arr);

void create_seq_swarm( double seed, double a , int q, int *arr, uthread_info_t *ti);

void create_seq_random_swarm( double seed, double a , int q, int *arr, uthread_info_t *ti);




static double R23, R46, T23, T46;
# 604 "swarm_isort64.comb.c"
static double randlc(double *X, double *A)
{
      static int KS;
      static double R23, R46, T23, T46;
      double T1, T2, T3, T4;
      double A1;
      double A2;
      double X1;
      double X2;
      double Z;
      int i, j;

      if (KS == 0)
      {
        R23 = 1.0;
        R46 = 1.0;
        T23 = 1.0;
        T46 = 1.0;

        for (i=1; i<=23; i++)
        {
          R23 = 0.50 * R23;
          T23 = 2.0 * T23;
        }
        for (i=1; i<=46; i++)
        {
          R46 = 0.50 * R46;
          T46 = 2.0 * T46;
        }
        KS = 1;
      }



      T1 = R23 * *A;
      j = (int)T1;
      A1 = j;
      A2 = *A - T23 * A1;





      T1 = R23 * *X;
      j = (int)T1;
      X1 = j;
      X2 = *X - T23 * X1;
      T1 = A1 * X2 + A2 * X1;

      j = (int)(R23 * T1);
      T2 = j;
      Z = T1 - T23 * T2;
      T3 = T23 * Z + A2 * X2;
      j = (int)(R46 * T3);
      T4 = j;
      *X = T3 - T46 * T4;
      return(R46 * *X);
}

static void init_nas() {

  int i;

  R23 = 1.0;
  R46 = 1.0;
  T23 = 1.0;
  T46 = 1.0;

  for (i=1; i<=23; i++) {
    R23 = 0.50 * R23;
    T23 = 2.0 * T23;
  }
  for (i=1; i<=46; i++) {
    R46 = 0.50 * R46;
    T46 = 2.0 * T46;
  }
}


static double randlc_swarm(double *X, double *A)
{
  double T1, T2, T3, T4;
  double A1;
  double A2;
  double X1;
  double X2;
  double Z;
  int j;






  T1 = R23 * *A;
  j = (int)T1;
  A1 = j;
  A2 = *A - T23 * A1;






  T1 = R23 * *X;
  j = (int)T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j = (int)(R23 * T1);
  T2 = j;
  Z = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j = (int)(R46 * T3);
  T4 = j;
  *X = T3 - T46 * T4;
  return(R46 * *X);
}
# 740 "swarm_isort64.comb.c"
double find_my_seed( long kn,
                       long np,
                       long nn,
                       double s,
                       double a )
{

  long i;

  double t1,t2,an;
  long mq,nq,kk,ik;



      nq = nn / np;

      for( mq=0; nq>1; mq++,nq/=2 )
          ;

      t1 = a;

      for( i=1; i<=mq; i++ )
        t2 = randlc( &t1, &t1 );

      an = t1;

      kk = kn;
      t1 = s;
      t2 = an;

      for( i=1; i<=100; i++ )
      {
        ik = kk / 2;
        if( 2 * ik != kk )
            randlc( &t1, &t2 );
        if( ik == 0 )
            break;
        randlc( &t2, &t2 );
        kk = ik;
      }

      return( t1 );

}
# 794 "swarm_isort64.comb.c"
void create_seq( double seed, double a , int q, int *arr)
{
 double x;
 register int i, k;

        k = (1<<19)/4;

 for (i=0; i<q; i++)
 {
     x = randlc(&seed, &a);
     x += randlc(&seed, &a);
         x += randlc(&seed, &a);
     x += randlc(&seed, &a);

            arr[i] = (int)(k*x);
 }
}



void create_seq_swarm( double seed, double a , int q, int *arr, uthread_info_t *ti)
{
 double x;
 register int i, k;

        k = (1<<19)/4;

 if ((ti->mythread) == 0)
   init_nas();
 SWARM_Barrier_sync(ti);

 for (i=0; i<q; i++)
 {
     x = randlc_swarm(&seed, &a);
     x += randlc_swarm(&seed, &a);
         x += randlc_swarm(&seed, &a);
     x += randlc_swarm(&seed, &a);

            arr[i] = (int)(k*x);
 }
}


void create_seq_random_swarm( double seed, double a , int q, int *arr,
      uthread_info_t *ti)
{
 register int i, k;

        k = 2147483648;

 if ((ti->mythread) == 0)
   init_nas();
 SWARM_Barrier_sync(ti);

 for (i=0; i<q; i++)
   arr[i] = (int)(k * randlc_swarm(&seed, &a));
}





void create_input_random_swarm(int myN, int *x, uthread_info_t *ti) {
  create_seq_random_swarm( 317*((ti->mythread)+17),
       1220703125.00,
       myN,
       x,
       ti);
}


void create_input_nas_swarm(int n, int *x, uthread_info_t *ti) {
  register int tsize, mynum, thtot;

  tsize = n / THREADS;
  mynum = (ti->mythread);
  thtot = THREADS;

  create_seq_swarm( find_my_seed( mynum,
       thtot,
       (n >> 2),
       314159265.00,
       1220703125.00),
       1220703125.00,
       tsize,
       x+(tsize*(ti->mythread)),
       ti);

}
# 894 "swarm_isort64.comb.c"
int MAXTHREADS = 64;


int THREADS;
uthread_info_t *uthread_info;
static pthread_t *spawn_thread;

static int _swarm_init=0;



static int _SWARM_bcast_i;
static long _SWARM_bcast_l;
static double _SWARM_bcast_d;
static char _SWARM_bcast_c;
static int *_SWARM_bcast_ip;
static long *_SWARM_bcast_lp;
static double *_SWARM_bcast_dp;
static char *_SWARM_bcast_cp;



static _SWARM_MULTICORE_barrier_t nbar;

int SWARM_mutex_init(SWARM_mutex_t **mutex, const SWARM_mutexattr_t *attr, uthread_info_t *ti)
{
  int r;
  r = 0;
  *mutex = (SWARM_mutex_t *)SWARM_malloc(sizeof(SWARM_mutex_t), ti);
  if ((ti->mythread) == 0) {
    r = pthread_mutex_init(*mutex, attr);
  }
  r = SWARM_Bcast_i(r, ti);
  return r;
}

int SWARM_mutex_destroy(SWARM_mutex_t *mutex, uthread_info_t *ti) {
  int r;
  r = 0;
  SWARM_Barrier_sync(ti);
  if ((ti->mythread) == 0) {
    r = pthread_mutex_destroy(mutex);
    free (mutex);
  }
  r = SWARM_Bcast_i(r, ti);
  return r;
}

static void SWARM_Barrier_sync_init(void) {
  nbar = _SWARM_MULTICORE_barrier_init(THREADS);
}

static void SWARM_Barrier_sync_destroy(void) {
  _SWARM_MULTICORE_barrier_destroy(nbar);
}

void SWARM_Barrier_sync(uthread_info_t *ti) {



  _SWARM_MULTICORE_barrier_wait(nbar);
}

static volatile int up_buf[((64)<<7)][2];
static volatile int down_buf[((64)<<7)];

static void
SWARM_Barrier_tree_init(void) {
  int i;

  for (i=0 ; i<THREADS ; i++)
    up_buf[((i)<<7)][0] = up_buf[((i)<<7)][1] = down_buf[((i)<<7)] = 0;
  return;
}

static void
SWARM_Barrier_tree_destroy(void) { return; }

static void
SWARM_Barrier_tree_up(uthread_info_t *ti) {

  register int myidx = (ti->mythread);
  register int parent = ((ti->mythread) - 1) / 2;
  register int odd_child = 2 * (ti->mythread) + 1;
  register int even_child = 2 * (ti->mythread) + 2;
  register int parity = (ti->mythread) & 1;

  if ((ti->mythread) == 0) {
    if (THREADS > 2) {
      while (up_buf[((myidx)<<7)][0] == 0 ||
      up_buf[((myidx)<<7)][1] == 0) ;
    }
    else if (THREADS == 2) {
 while (up_buf[((myidx)<<7)][1] == 0) ;
    }
  }
  else
    if (odd_child >= THREADS)
      up_buf[((parent)<<7)][parity]++;
    else
      if (even_child >= THREADS) {
 while (up_buf[((myidx)<<7)][1] == 0) ;
 up_buf[((parent)<<7)][parity]++;
      }
      else {
 while (up_buf[((myidx)<<7)][0] == 0 ||
        up_buf[((myidx)<<7)][1] == 0) ;
 up_buf[((parent)<<7)][parity]++;
      }

  up_buf[((myidx)<<7)][0] = up_buf[((myidx)<<7)][1] = 0;



  return;
}

static void
SWARM_Barrier_tree_down(uthread_info_t *ti) {

  register int myidx = (ti->mythread);
  register int left = 2 * (ti->mythread) + 1;
  register int right = 2 * (ti->mythread) + 2;

  if ((ti->mythread) != 0)
    while (down_buf[((myidx)<<7)] == 0) ;

  if (left < THREADS)
    down_buf[((left)<<7)]++;
  if (right < THREADS)
    down_buf[((right)<<7)]++;

  down_buf[((myidx)<<7)] = 0;



  return;
}

void
SWARM_Barrier_tree(uthread_info_t *ti) {
  SWARM_Barrier_tree_up(ti);
  SWARM_Barrier_tree_down(ti);
  return;
}

void *SWARM_malloc(int bytes, uthread_info_t *ti) {
  void *ptr;
  ptr=((void *)0);
  if ((ti->mythread) == 0) {
    ptr = malloc(bytes);
    assert_malloc(ptr);
  }
  return(SWARM_Bcast_cp(ptr, ti));
}

void *SWARM_malloc_l(long bytes, uthread_info_t *ti) {
  void *ptr;
  ptr=((void *)0);
  if ((ti->mythread) == 0) {
    ptr = malloc(bytes);
    assert_malloc(ptr);
  }
  return(SWARM_Bcast_cp(ptr, ti));
}

void SWARM_free(void *ptr, uthread_info_t *ti) {
  if ((ti->mythread) == 0) {



    free(ptr);

  }
}

int SWARM_Bcast_i(int myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_i = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_i);
}

long SWARM_Bcast_l(long myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_l = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_l);
}

double SWARM_Bcast_d(double myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_d = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_d);
}

char SWARM_Bcast_c(char myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_c = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_c);
}

int *SWARM_Bcast_ip(int *myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_ip = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_ip);
}

long *SWARM_Bcast_lp(long *myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_lp = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_lp);
}

double *SWARM_Bcast_dp(double *myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_dp = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_dp);
}

char *SWARM_Bcast_cp(char *myval, uthread_info_t *ti) {

  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) {
    _SWARM_bcast_cp = myval;
  }

  SWARM_Barrier_sync(ti);
  return (_SWARM_bcast_cp);
}

static _SWARM_MULTICORE_reduce_d_t red_d;

double SWARM_Reduce_d(double myval, reduce_t op, uthread_info_t *ti) {
  return (_SWARM_MULTICORE_reduce_d(red_d, myval, op));
}


static _SWARM_MULTICORE_reduce_i_t red_i;

int SWARM_Reduce_i(int myval, reduce_t op, uthread_info_t *ti) {
  return (_SWARM_MULTICORE_reduce_i(red_i,myval,op));
}

static _SWARM_MULTICORE_reduce_l_t red_l;

long SWARM_Reduce_l(long myval, reduce_t op, uthread_info_t *ti) {
  return (_SWARM_MULTICORE_reduce_l(red_l,myval,op));
}

static _SWARM_MULTICORE_scan_i_t scan_i;

int SWARM_Scan_i(int myval, reduce_t op, uthread_info_t *ti) {
  return(_SWARM_MULTICORE_scan_i(scan_i,myval,op,(ti->mythread)));
}

static _SWARM_MULTICORE_scan_l_t scan_l;

long SWARM_Scan_l(long myval, reduce_t op, uthread_info_t *ti) {
  return(_SWARM_MULTICORE_scan_l(scan_l,myval,op,(ti->mythread)));
}

static _SWARM_MULTICORE_scan_d_t scan_d;

double SWARM_Scan_d(double myval, reduce_t op, uthread_info_t *ti) {
  return(_SWARM_MULTICORE_scan_d(scan_d,myval,op,(ti->mythread)));
}

static void SWARM_print_help(char **argv)
{
     printf ("SWARM usage: %s [-t #threads] [-o outfile]\n", argv[0]);
     printf ("\t-t #threads    overrides the default number of threads\n");
     printf ("\t-o outfile     redirects standard output to outfile\n");
}



static void SWARM_error (int lineno, const char *func, const char *format, ...)
{
    char msg[512];

    va_list args;
    __builtin_va_start(args,format);
    vsprintf(msg, format, args);

    fprintf (stderr, "SWARM_%s (line %d): %s\n", func, lineno, msg);
    fflush (stderr);

    SWARM_Finalize();

    exit (-1);

}


FILE *SWARM_outfile;
static char *SWARM_outfilename;
# 1247 "swarm_isort64.comb.c"
static int SWARM_read_arguments (int argc, char **argv)
{
     extern char *optarg;
     extern int optind;
     char *tail;
     int c, i;

     if (argv[0] == ((void *)0))
   SWARM_error (1255, "SWARM_read_arguments",
         "invalid argument array");

     if (argc < 1) return 0;

     opterr = 0;
     while ((c = getopt (argc, argv, "ht:o:")) != -1)
     {
   switch (c)
   {

   case 't':
        i = (int)strtol (optarg, &tail, 0);
        if (optarg == tail)
      SWARM_error (1269, "read_arguments",
     "invalid argument %s to option -t", optarg);
        if (i <= 0)
      SWARM_error (1272, "read_arguments",
     "# of threads must be greater than zero");
        else
      THREADS = i;
        break;

   case 'o':
        SWARM_outfilename = strdup(optarg);
        if ((SWARM_outfile = fopen (SWARM_outfilename, "w")) == ((void *)0))
      SWARM_error (1281, "read_arguments",
     "unable to open outfile %s", SWARM_outfilename);
        break;

   case 'h':
        SWARM_print_help(argv);
        exit(0);
        break;

   default:
  SWARM_error (1291, "read_arguments",
        "`%c': unrecognized argument", c);

                break;
    }
      }


     if (argv[optind] != ((void *)0)) return optind;
     else return 0;
}



static void
SWARM_get_args(int *argc, char* **argv) {
  int numarg = *argc;
  int done = 0;
  char
    *s,**argvv = *argv;
  char
    *outfilename = ((void *)0);

  SWARM_outfile = stdout;

  while ((--numarg > 0) && !done)
    if ((*++argvv)[0] == '-')
      for (s=argvv[0]+1; *s != '\0'; s++) {
 if (*s == '-')
   done = 1;
 else {
   switch (*s) {
   case 'o':
     if (numarg <= 1)
       perror("output filename expected after -o (e.g. -o filename)");
     numarg--;
     outfilename = (char *)malloc(80*sizeof(char));
     strcpy(outfilename, *++argvv);
     SWARM_outfile = fopen(outfilename,"a+");
     break;
   case 't':
     if (numarg <= 1)
       perror("number of threads per node expected after -t");
     numarg--;
     THREADS = atoi(*++argvv);

     break;
   case 'h':
     fprintf(SWARM_outfile,"SWARM Options:\n");
     fprintf(SWARM_outfile," -t <number of threads per node>\n");
     fprintf(SWARM_outfile,"\n\n");
     fflush(SWARM_outfile);
     break;

   }
 }
      }
  if (done) {
    *argc = numarg;
    *argv = ++argvv;
  }
  else {
    *argc = 0;
    *argv = ((void *)0);
  }

  return;
}
# 1390 "swarm_isort64.comb.c"
static int SWARM_get_num_cores(void)
{
 int num_cores = 2;
# 1403 "swarm_isort64.comb.c"
      num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);



 return num_cores;
}






void SWARM_Init(int *argc, char* **argv)
{

 int i;



    uthread_info_t *ti;
    int moreargs;





    THREADS = SWARM_get_num_cores();

    SWARM_outfile = stdout;
    SWARM_outfilename = ((void *)0);

    moreargs = SWARM_read_arguments (*argc, *argv);


    fprintf(SWARM_outfile,"THREADS: %d\n", THREADS);
    fflush(SWARM_outfile);






    SWARM_Barrier_sync_init();
    SWARM_Barrier_tree_init();

    red_i = _SWARM_MULTICORE_reduce_init_i(THREADS);
    red_l = _SWARM_MULTICORE_reduce_init_l(THREADS);
    red_d = _SWARM_MULTICORE_reduce_init_d(THREADS);

    scan_i = _SWARM_MULTICORE_scan_init_i(THREADS);
    scan_l = _SWARM_MULTICORE_scan_init_l(THREADS);
    scan_d = _SWARM_MULTICORE_scan_init_d(THREADS);
# 1487 "swarm_isort64.comb.c"
    spawn_thread = (pthread_t *)malloc(((THREADS)<<7)*
           sizeof(pthread_t));
    assert_malloc(spawn_thread);
    uthread_info = (uthread_info_t *)malloc(((THREADS)<<7)*
       sizeof(uthread_info_t));
    assert_malloc(uthread_info);

    ti = uthread_info;

    for (i=0 ; i<THREADS ; i++) {
      ti->mythread = i;

      if (moreargs > 0)
      {
    ti->argc = (*argc)-moreargs;
    ti->argv = (*argv)+moreargs;
      }
      else
      {
    ti->argc = 0;
    ti->argv = (char **)((void *)0);
      }
# 1517 "swarm_isort64.comb.c"
      ti += ((1)<<7);
    }

    _swarm_init=1;
}

void SWARM_Run(void *start_routine)
{
     int i, rc;
     int *parg;
     uthread_info_t *ti;
     void *(*f)(void *);

     f = (void *(*)(void *))start_routine;

     if (!_swarm_init)
     {
   fprintf(stderr,"ERROR: SWARM_Init() not called\n");
   perror("SWARM_Run cannot call SWARM_Init()");
     }

     ti = uthread_info;

     for (i=0 ; i<THREADS ; i++)
     {

   rc = pthread_create(spawn_thread + ((i)<<7),



         ((void *)0),

         f,
         ti);

   if (rc != 0)
   {
        switch (rc)
        {
        case 11:
      SWARM_error (1557, "Run:pthread_create",
     "not enough resources to create another thread");
      break;

        case 22:
      SWARM_error (1562, "Run:pthread_create",
     "invalid thread attributes");
      break;

        case 1:
      SWARM_error (1567, "Run:pthread_create",
     "insufficient permissions for setting scheduling parameters or policy ");
      break;

        default:
      SWARM_error (1572, "Run:pthread_create", "error code %d", rc);
        }
   }

   ti += ((1)<<7);
     }

     for (i=0 ; i<THREADS ; i++)
     {
   rc = pthread_join(spawn_thread[((i)<<7)], (void *)&parg);
   if (rc != 0)
   {
        switch (rc)
        {
        case 22:
      SWARM_error (1587, "Run:pthread_join", "specified thread is not joinable");
      break;

        case 3:
      SWARM_error (1591, "Run:pthread_join", "cannot find thread");
      break;

        default:
      SWARM_error (1595, "Run:pthread_join", "error code %d", rc);

        }
   }
     }
}

void SWARM_Finalize(void)
{





     _SWARM_MULTICORE_reduce_destroy_i(red_i);
     _SWARM_MULTICORE_reduce_destroy_l(red_l);
     _SWARM_MULTICORE_reduce_destroy_d(red_d);
     _SWARM_MULTICORE_scan_destroy_i(scan_i);
     _SWARM_MULTICORE_scan_destroy_l(scan_l);
     _SWARM_MULTICORE_scan_destroy_d(scan_d);

     SWARM_Barrier_sync_destroy();
     SWARM_Barrier_tree_destroy();

     free(uthread_info);
     free(spawn_thread);

     if (SWARM_outfile != ((void *)0))
   fclose(SWARM_outfile);
     if (SWARM_outfilename != ((void *)0))
   free(SWARM_outfilename);
}

void SWARM_Cleanup(uthread_info_t *ti)
{



     return;
}

void assert_malloc(void *ptr)
{
 if (ptr==((void *)0))
     perror("ERROR: assert_malloc");
}

double get_seconds(void)
{
 struct timeval t;
   struct timezone z;
   gettimeofday(&t,&z);
   return (double)t.tv_sec+((double)t.tv_usec/(double)1000000.0);
}
# 1687 "swarm_isort64.comb.c"
_SWARM_MULTICORE_barrier_t _SWARM_MULTICORE_barrier_init(int n_clients) {
  _SWARM_MULTICORE_barrier_t nbarrier = (_SWARM_MULTICORE_barrier_t)
    malloc(sizeof(struct _SWARM_MULTICORE_barrier));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_barrier_destroy(_SWARM_MULTICORE_barrier_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier);
}

void _SWARM_MULTICORE_barrier_wait(_SWARM_MULTICORE_barrier_t nbarrier) {
  int my_phase;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
}

_SWARM_MULTICORE_reduce_i_t _SWARM_MULTICORE_reduce_init_i(int n_clients) {
  _SWARM_MULTICORE_reduce_i_t nbarrier = (_SWARM_MULTICORE_reduce_i_t)
    malloc(sizeof(struct _SWARM_MULTICORE_reduce_i_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->sum = 0;
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_reduce_destroy_i(_SWARM_MULTICORE_reduce_i_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier);
}

int _SWARM_MULTICORE_reduce_i(_SWARM_MULTICORE_reduce_i_t nbarrier, int val, reduce_t op) {
  int my_phase;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  if (nbarrier->n_waiting==0) {
    nbarrier->sum = val;
  }
  else {
    switch (op) {
    case MIN: nbarrier->sum = ((nbarrier->sum) < (val) ? (nbarrier->sum) : (val)); break;
    case MAX : nbarrier->sum = ((nbarrier->sum) > (val) ? (nbarrier->sum) : (val)); break;
    case SUM : nbarrier->sum += val; break;
    default : perror("ERROR: _SWARM_MULTICORE_reduce_i() Bad reduction operator");
    }
  }
  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    nbarrier->result = nbarrier->sum;
    nbarrier->sum = 0;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result);
}

_SWARM_MULTICORE_reduce_l_t _SWARM_MULTICORE_reduce_init_l(int n_clients) {
  _SWARM_MULTICORE_reduce_l_t nbarrier = (_SWARM_MULTICORE_reduce_l_t)
    malloc(sizeof(struct _SWARM_MULTICORE_reduce_l_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->sum = 0;
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_reduce_destroy_l(_SWARM_MULTICORE_reduce_l_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier);
}

long _SWARM_MULTICORE_reduce_l(_SWARM_MULTICORE_reduce_l_t nbarrier, long val, reduce_t op) {
  int my_phase;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  if (nbarrier->n_waiting==0) {
    nbarrier->sum = val;
  }
  else {
    switch (op) {
    case MIN: nbarrier->sum = ((nbarrier->sum) < (val) ? (nbarrier->sum) : (val)); break;
    case MAX : nbarrier->sum = ((nbarrier->sum) > (val) ? (nbarrier->sum) : (val)); break;
    case SUM : nbarrier->sum += val; break;
    default : perror("ERROR: _SWARM_MULTICORE_reduce_l() Bad reduction operator");
    }
  }
  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    nbarrier->result = nbarrier->sum;
    nbarrier->sum = 0;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result);
}

_SWARM_MULTICORE_reduce_d_t _SWARM_MULTICORE_reduce_init_d(int n_clients) {
  _SWARM_MULTICORE_reduce_d_t nbarrier = (_SWARM_MULTICORE_reduce_d_t)
    malloc(sizeof(struct _SWARM_MULTICORE_reduce_d_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->sum = 0.000001;
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_reduce_destroy_d(_SWARM_MULTICORE_reduce_d_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier);
}

double _SWARM_MULTICORE_reduce_d(_SWARM_MULTICORE_reduce_d_t nbarrier, double val, reduce_t op) {
  int my_phase;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  if (nbarrier->n_waiting==0) {
    nbarrier->sum = val;
  }
  else {
    switch (op) {
    case MIN: nbarrier->sum = ((nbarrier->sum) < (val) ? (nbarrier->sum) : (val)); break;
    case MAX : nbarrier->sum = ((nbarrier->sum) > (val) ? (nbarrier->sum) : (val)); break;
    case SUM : nbarrier->sum += val; break;
    default : perror("ERROR: _SWARM_MULTICORE_reduce_i() Bad reduction operator");
    }
  }
  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    nbarrier->result = nbarrier->sum;
    nbarrier->sum = 0.0;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result);
}

_SWARM_MULTICORE_scan_i_t _SWARM_MULTICORE_scan_init_i(int n_clients) {
  _SWARM_MULTICORE_scan_i_t nbarrier = (_SWARM_MULTICORE_scan_i_t)
    malloc(sizeof(struct _SWARM_MULTICORE_scan_i_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->result = (int *)malloc(n_clients*sizeof(int));
    assert_malloc(nbarrier->result);
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_scan_destroy_i(_SWARM_MULTICORE_scan_i_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier->result);
  free(nbarrier);
}

int _SWARM_MULTICORE_scan_i(_SWARM_MULTICORE_scan_i_t nbarrier, int val, reduce_t op,int th_index) {
  int my_phase,i,temp;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  nbarrier->result[th_index] = val;

  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    switch (op) {
    case MIN : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) < (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case MAX : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) > (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case SUM :
      for(i = 1; i < nbarrier->n_clients;i++)
         nbarrier->result[i] += nbarrier->result[i-1];
      break;
    default : perror("ERROR: _SWARM_MULTICORE_scan_i() Bad reduction operator");
    }
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result[th_index]);
}

_SWARM_MULTICORE_scan_l_t _SWARM_MULTICORE_scan_init_l(int n_clients) {
  _SWARM_MULTICORE_scan_l_t nbarrier = (_SWARM_MULTICORE_scan_l_t)
    malloc(sizeof(struct _SWARM_MULTICORE_scan_l_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->result = (long *)malloc(n_clients*sizeof(long));
    assert_malloc(nbarrier->result);
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_scan_destroy_l(_SWARM_MULTICORE_scan_l_t nbarrier) {
  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier->result);
  free(nbarrier);
}

long _SWARM_MULTICORE_scan_l(_SWARM_MULTICORE_scan_l_t nbarrier, long val, reduce_t op, int th_index) {
  int my_phase,i;
  long temp;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  nbarrier->result[th_index] = val;

  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    switch (op) {
    case MIN : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) < (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case MAX : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) > (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case SUM :
      for(i = 1; i < nbarrier->n_clients;i++)
         nbarrier->result[i] += nbarrier->result[i-1];
      break;
    default : perror("ERROR: _SWARM_MULTICORE_scan_i() Bad reduction operator");
    }
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result[th_index]);
}

_SWARM_MULTICORE_scan_d_t _SWARM_MULTICORE_scan_init_d(int n_clients) {
  _SWARM_MULTICORE_scan_d_t nbarrier = (_SWARM_MULTICORE_scan_d_t)
    malloc(sizeof(struct _SWARM_MULTICORE_scan_d_s));
  assert_malloc(nbarrier);

  if (nbarrier != ((void *)0)) {
    nbarrier->n_clients = n_clients;
    nbarrier->n_waiting = 0;
    nbarrier->phase = 0;
    nbarrier->result = (double *)malloc(n_clients*sizeof(double));
    assert_malloc(nbarrier->result);
    pthread_mutex_init(&nbarrier->lock, ((void *)0));
    pthread_cond_init(&nbarrier->wait_cv, ((void *)0));
  }
  return(nbarrier);
}

void _SWARM_MULTICORE_scan_destroy_d(_SWARM_MULTICORE_scan_d_t nbarrier) {

  pthread_mutex_destroy(&nbarrier->lock);
  pthread_cond_destroy(&nbarrier->wait_cv);
  free(nbarrier->result);
  free(nbarrier);
}

double _SWARM_MULTICORE_scan_d(_SWARM_MULTICORE_scan_d_t nbarrier, double val, reduce_t op,int th_index) {
  int my_phase,i;
  double temp;

  pthread_mutex_lock(&nbarrier->lock);
  my_phase = nbarrier->phase;
  nbarrier->result[th_index] = val;
  nbarrier->n_waiting++;
  if (nbarrier->n_waiting == nbarrier->n_clients) {
    switch (op) {
    case MIN : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) < (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case MAX : temp = nbarrier->result[0];
      for(i = 1; i < nbarrier->n_clients;i++) {
         temp = ((nbarrier->result[i]) > (temp) ? (nbarrier->result[i]) : (temp));
         nbarrier->result[i] = temp;
      }
      break;
    case SUM :
      for(i = 1; i < nbarrier->n_clients;i++)
         nbarrier->result[i] += nbarrier->result[i-1];
      break;
    default : perror("ERROR: _SWARM_MULTICORE_scan_i() Bad reduction operator");
    }
    nbarrier->n_waiting = 0;
    nbarrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&nbarrier->wait_cv);
  }
  while (nbarrier->phase == my_phase) {
    pthread_cond_wait(&nbarrier->wait_cv, &nbarrier->lock);
  }
  pthread_mutex_unlock(&nbarrier->lock);
  return(nbarrier->result[th_index]);
}



_SWARM_MULTICORE_spin_barrier_t _SWARM_MULTICORE_spin_barrier_init(int n_clients) {
  _SWARM_MULTICORE_spin_barrier_t sbarrier = (_SWARM_MULTICORE_spin_barrier_t)
    malloc(sizeof(struct _SWARM_MULTICORE_spin_barrier));
  assert_malloc(sbarrier);

  if (sbarrier != ((void *)0)) {
    sbarrier->n_clients = n_clients;
    sbarrier->n_waiting = 0;
    sbarrier->phase = 0;
    pthread_mutex_init(&sbarrier->lock, ((void *)0));
  }
  return(sbarrier);
}

void _SWARM_MULTICORE_spin_barrier_destroy(_SWARM_MULTICORE_spin_barrier_t sbarrier) {
  pthread_mutex_destroy(&sbarrier->lock);
  free(sbarrier);
}

void _SWARM_MULTICORE_spin_barrier_wait(_SWARM_MULTICORE_spin_barrier_t sbarrier) {
  int my_phase;

  while (pthread_mutex_trylock(&sbarrier->lock) == 16) ;
  my_phase = sbarrier->phase;
  sbarrier->n_waiting++;
  if (sbarrier->n_waiting == sbarrier->n_clients) {
    sbarrier->n_waiting = 0;
    sbarrier->phase = 1 - my_phase;
  }
  pthread_mutex_unlock(&sbarrier->lock);

  while (sbarrier->phase == my_phase) ;
}
# 2133 "swarm_isort64.comb.c"
static void test_radixsort_swarm(long N1, uthread_info_t *ti)
{
  int *inArr, *outArr;


  double secs, tsec;


  inArr = (int *)SWARM_malloc_l(N1 * sizeof(int), ti);
  outArr = (int *)SWARM_malloc_l(N1 * sizeof(int), ti);

  create_input_nas_swarm(N1, inArr, ti);


  SWARM_Barrier_sync(ti);
  secs = get_seconds();


  radixsort_swarm_s3(N1,inArr,outArr,ti);


  secs = get_seconds() - secs;
  secs = ((secs) > (0.000001) ? (secs) : (0.000001));
  tsec = SWARM_Reduce_d(secs,MAX, ti);
  if ((ti->mythread) == 0) {
    fprintf(stdout,"T: %3d n: %13ld  SSort: %9.6lf  (MB:%5ld)\n",
     THREADS,N1,tsec,
     ((long)ceil(((double)N1*2*sizeof(int))/(double)(1<<20))));
    fflush(stdout);
  }


  SWARM_Barrier_sync(ti);

  if ((ti->mythread) == 0) radixsort_check(N1, outArr);

  SWARM_Barrier_sync(ti);

  SWARM_free(outArr, ti);
  SWARM_free(inArr, ti);
}



static void *swarmtest(uthread_info_t *ti)
{
  long i;
# 2188 "swarm_isort64.comb.c"
  SWARM_Barrier_sync(ti);


  i = (long)1 << 24;
  test_radixsort_swarm(i, ti);

  SWARM_Barrier_sync(ti);





  SWARM_Cleanup(ti); pthread_exit((((void *)0))); return 0;;
}

int main(int argc, char **argv)
{
  SWARM_Init(&argc,&argv);
  SWARM_Run((void *)swarmtest);
  SWARM_Finalize();
  return 0;
}
