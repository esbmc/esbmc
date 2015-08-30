# 1 "exStbKey.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "exStbKey.c"
# 58 "exStbKey.c"
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

# 59 "exStbKey.c" 2
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
# 100 "/usr/include/sys/types.h" 3
typedef __pid_t pid_t;




typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 133 "/usr/include/sys/types.h" 3
# 1 "/usr/include/time.h" 1 3
# 75 "/usr/include/time.h" 3


typedef __time_t time_t;



# 93 "/usr/include/time.h" 3
typedef __clockid_t clockid_t;
# 105 "/usr/include/time.h" 3
typedef __timer_t timer_t;
# 134 "/usr/include/sys/types.h" 2 3
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





# 1 "/usr/include/time.h" 1 3
# 121 "/usr/include/time.h" 3
struct timespec
  {
    __time_t tv_sec;
    long int tv_nsec;
  };
# 45 "/usr/include/sys/select.h" 2 3

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

# 60 "exStbKey.c" 2
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

# 61 "exStbKey.c" 2
# 74 "exStbKey.c"
# 1 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h" 1
# 32 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
# 1 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/dfb_types.h" 1
# 49 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/dfb_types.h"
# 1 "/usr/include/stdint.h" 1 3
# 27 "/usr/include/stdint.h" 3
# 1 "/usr/include/bits/wchar.h" 1 3
# 28 "/usr/include/stdint.h" 2 3
# 1 "/usr/include/bits/wordsize.h" 1 3
# 29 "/usr/include/stdint.h" 2 3
# 49 "/usr/include/stdint.h" 3
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
# 126 "/usr/include/stdint.h" 3
typedef int intptr_t;


typedef unsigned int uintptr_t;
# 138 "/usr/include/stdint.h" 3
__extension__
typedef long long int intmax_t;
__extension__
typedef unsigned long long int uintmax_t;
# 50 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/dfb_types.h" 2

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
# 33 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h" 2
# 1 "/usr/include/sys/time.h" 1 3
# 27 "/usr/include/sys/time.h" 3
# 1 "/usr/include/time.h" 1 3
# 28 "/usr/include/sys/time.h" 2 3

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

# 34 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h" 2

# 1 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h" 1
# 41 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h"
typedef enum {
     DIKT_UNICODE = 0x0000,

     DIKT_SPECIAL = 0xF000,
     DIKT_FUNCTION = 0xF100,
     DIKT_MODIFIER = 0xF200,
     DIKT_LOCK = 0xF300,
     DIKT_DEAD = 0xF400,
     DIKT_CUSTOM = 0xF500,
     DIKT_IDENTIFIER = 0xF600
} DFBInputDeviceKeyType;
# 72 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb_keyboard.h"
typedef enum {
     DIMKI_SHIFT,
     DIMKI_CONTROL,
     DIMKI_ALT,
     DIMKI_ALTGR,
     DIMKI_META,
     DIMKI_SUPER,
     DIMKI_HYPER,

     DIMKI_FIRST = DIMKI_SHIFT,
     DIMKI_LAST = DIMKI_HYPER
} DFBInputDeviceModifierKeyIdentifier;




typedef enum {
     DIKI_UNKNOWN = ((DIKT_IDENTIFIER) | (0)),

     DIKI_A,
     DIKI_B,
     DIKI_C,
     DIKI_D,
     DIKI_E,
     DIKI_F,
     DIKI_G,
     DIKI_H,
     DIKI_I,
     DIKI_J,
     DIKI_K,
     DIKI_L,
     DIKI_M,
     DIKI_N,
     DIKI_O,
     DIKI_P,
     DIKI_Q,
     DIKI_R,
     DIKI_S,
     DIKI_T,
     DIKI_U,
     DIKI_V,
     DIKI_W,
     DIKI_X,
     DIKI_Y,
     DIKI_Z,

     DIKI_0,
     DIKI_1,
     DIKI_2,
     DIKI_3,
     DIKI_4,
     DIKI_5,
     DIKI_6,
     DIKI_7,
     DIKI_8,
     DIKI_9,

     DIKI_F1,
     DIKI_F2,
     DIKI_F3,
     DIKI_F4,
     DIKI_F5,
     DIKI_F6,
     DIKI_F7,
     DIKI_F8,
     DIKI_F9,
     DIKI_F10,
     DIKI_F11,
     DIKI_F12,

     DIKI_SHIFT_L,
     DIKI_SHIFT_R,
     DIKI_CONTROL_L,
     DIKI_CONTROL_R,
     DIKI_ALT_L,
     DIKI_ALT_R,
     DIKI_META_L,
     DIKI_META_R,
     DIKI_SUPER_L,
     DIKI_SUPER_R,
     DIKI_HYPER_L,
     DIKI_HYPER_R,

     DIKI_CAPS_LOCK,
     DIKI_NUM_LOCK,
     DIKI_SCROLL_LOCK,

     DIKI_ESCAPE,
     DIKI_LEFT,
     DIKI_RIGHT,
     DIKI_UP,
     DIKI_DOWN,
     DIKI_TAB,
     DIKI_ENTER,
     DIKI_SPACE,
     DIKI_BACKSPACE,
     DIKI_INSERT,
     DIKI_DELETE,
     DIKI_HOME,
     DIKI_END,
     DIKI_PAGE_UP,
     DIKI_PAGE_DOWN,
     DIKI_PRINT,
     DIKI_PAUSE,





     DIKI_QUOTE_LEFT,
     DIKI_MINUS_SIGN,
     DIKI_EQUALS_SIGN,
     DIKI_BRACKET_LEFT,
     DIKI_BRACKET_RIGHT,
     DIKI_BACKSLASH,
     DIKI_SEMICOLON,
     DIKI_QUOTE_RIGHT,
     DIKI_COMMA,
     DIKI_PERIOD,
     DIKI_SLASH,

     DIKI_LESS_SIGN,

     DIKI_KP_DIV,
     DIKI_KP_MULT,
     DIKI_KP_MINUS,
     DIKI_KP_PLUS,
     DIKI_KP_ENTER,
     DIKI_KP_SPACE,
     DIKI_KP_TAB,
     DIKI_KP_F1,
     DIKI_KP_F2,
     DIKI_KP_F3,
     DIKI_KP_F4,
     DIKI_KP_EQUAL,
     DIKI_KP_SEPARATOR,

     DIKI_KP_DECIMAL,
     DIKI_KP_0,
     DIKI_KP_1,
     DIKI_KP_2,
     DIKI_KP_3,
     DIKI_KP_4,
     DIKI_KP_5,
     DIKI_KP_6,
     DIKI_KP_7,
     DIKI_KP_8,
     DIKI_KP_9,

     DIKI_KEYDEF_END,
     DIKI_NUMBER_OF_KEYS = DIKI_KEYDEF_END - ((DIKT_IDENTIFIER) | (0))

} DFBInputDeviceKeyIdentifier;




typedef enum {






     DIKS_NULL = ((DIKT_UNICODE) | (0x00)),
     DIKS_BACKSPACE = ((DIKT_UNICODE) | (0x08)),
     DIKS_TAB = ((DIKT_UNICODE) | (0x09)),
     DIKS_RETURN = ((DIKT_UNICODE) | (0x0D)),
     DIKS_CANCEL = ((DIKT_UNICODE) | (0x18)),
     DIKS_ESCAPE = ((DIKT_UNICODE) | (0x1B)),
     DIKS_SPACE = ((DIKT_UNICODE) | (0x20)),
     DIKS_EXCLAMATION_MARK = ((DIKT_UNICODE) | (0x21)),
     DIKS_QUOTATION = ((DIKT_UNICODE) | (0x22)),
     DIKS_NUMBER_SIGN = ((DIKT_UNICODE) | (0x23)),
     DIKS_DOLLAR_SIGN = ((DIKT_UNICODE) | (0x24)),
     DIKS_PERCENT_SIGN = ((DIKT_UNICODE) | (0x25)),
     DIKS_AMPERSAND = ((DIKT_UNICODE) | (0x26)),
     DIKS_APOSTROPHE = ((DIKT_UNICODE) | (0x27)),
     DIKS_PARENTHESIS_LEFT = ((DIKT_UNICODE) | (0x28)),
     DIKS_PARENTHESIS_RIGHT = ((DIKT_UNICODE) | (0x29)),
     DIKS_ASTERISK = ((DIKT_UNICODE) | (0x2A)),
     DIKS_PLUS_SIGN = ((DIKT_UNICODE) | (0x2B)),
     DIKS_COMMA = ((DIKT_UNICODE) | (0x2C)),
     DIKS_MINUS_SIGN = ((DIKT_UNICODE) | (0x2D)),
     DIKS_PERIOD = ((DIKT_UNICODE) | (0x2E)),
     DIKS_SLASH = ((DIKT_UNICODE) | (0x2F)),
     DIKS_0 = ((DIKT_UNICODE) | (0x30)),
     DIKS_1 = ((DIKT_UNICODE) | (0x31)),
     DIKS_2 = ((DIKT_UNICODE) | (0x32)),
     DIKS_3 = ((DIKT_UNICODE) | (0x33)),
     DIKS_4 = ((DIKT_UNICODE) | (0x34)),
     DIKS_5 = ((DIKT_UNICODE) | (0x35)),
     DIKS_6 = ((DIKT_UNICODE) | (0x36)),
     DIKS_7 = ((DIKT_UNICODE) | (0x37)),
     DIKS_8 = ((DIKT_UNICODE) | (0x38)),
     DIKS_9 = ((DIKT_UNICODE) | (0x39)),
     DIKS_COLON = ((DIKT_UNICODE) | (0x3A)),
     DIKS_SEMICOLON = ((DIKT_UNICODE) | (0x3B)),
     DIKS_LESS_THAN_SIGN = ((DIKT_UNICODE) | (0x3C)),
     DIKS_EQUALS_SIGN = ((DIKT_UNICODE) | (0x3D)),
     DIKS_GREATER_THAN_SIGN = ((DIKT_UNICODE) | (0x3E)),
     DIKS_QUESTION_MARK = ((DIKT_UNICODE) | (0x3F)),
     DIKS_AT = ((DIKT_UNICODE) | (0x40)),
     DIKS_CAPITAL_A = ((DIKT_UNICODE) | (0x41)),
     DIKS_CAPITAL_B = ((DIKT_UNICODE) | (0x42)),
     DIKS_CAPITAL_C = ((DIKT_UNICODE) | (0x43)),
     DIKS_CAPITAL_D = ((DIKT_UNICODE) | (0x44)),
     DIKS_CAPITAL_E = ((DIKT_UNICODE) | (0x45)),
     DIKS_CAPITAL_F = ((DIKT_UNICODE) | (0x46)),
     DIKS_CAPITAL_G = ((DIKT_UNICODE) | (0x47)),
     DIKS_CAPITAL_H = ((DIKT_UNICODE) | (0x48)),
     DIKS_CAPITAL_I = ((DIKT_UNICODE) | (0x49)),
     DIKS_CAPITAL_J = ((DIKT_UNICODE) | (0x4A)),
     DIKS_CAPITAL_K = ((DIKT_UNICODE) | (0x4B)),
     DIKS_CAPITAL_L = ((DIKT_UNICODE) | (0x4C)),
     DIKS_CAPITAL_M = ((DIKT_UNICODE) | (0x4D)),
     DIKS_CAPITAL_N = ((DIKT_UNICODE) | (0x4E)),
     DIKS_CAPITAL_O = ((DIKT_UNICODE) | (0x4F)),
     DIKS_CAPITAL_P = ((DIKT_UNICODE) | (0x50)),
     DIKS_CAPITAL_Q = ((DIKT_UNICODE) | (0x51)),
     DIKS_CAPITAL_R = ((DIKT_UNICODE) | (0x52)),
     DIKS_CAPITAL_S = ((DIKT_UNICODE) | (0x53)),
     DIKS_CAPITAL_T = ((DIKT_UNICODE) | (0x54)),
     DIKS_CAPITAL_U = ((DIKT_UNICODE) | (0x55)),
     DIKS_CAPITAL_V = ((DIKT_UNICODE) | (0x56)),
     DIKS_CAPITAL_W = ((DIKT_UNICODE) | (0x57)),
     DIKS_CAPITAL_X = ((DIKT_UNICODE) | (0x58)),
     DIKS_CAPITAL_Y = ((DIKT_UNICODE) | (0x59)),
     DIKS_CAPITAL_Z = ((DIKT_UNICODE) | (0x5A)),
     DIKS_SQUARE_BRACKET_LEFT = ((DIKT_UNICODE) | (0x5B)),
     DIKS_BACKSLASH = ((DIKT_UNICODE) | (0x5C)),
     DIKS_SQUARE_BRACKET_RIGHT = ((DIKT_UNICODE) | (0x5D)),
     DIKS_CIRCUMFLEX_ACCENT = ((DIKT_UNICODE) | (0x5E)),
     DIKS_UNDERSCORE = ((DIKT_UNICODE) | (0x5F)),
     DIKS_GRAVE_ACCENT = ((DIKT_UNICODE) | (0x60)),
     DIKS_SMALL_A = ((DIKT_UNICODE) | (0x61)),
     DIKS_SMALL_B = ((DIKT_UNICODE) | (0x62)),
     DIKS_SMALL_C = ((DIKT_UNICODE) | (0x63)),
     DIKS_SMALL_D = ((DIKT_UNICODE) | (0x64)),
     DIKS_SMALL_E = ((DIKT_UNICODE) | (0x65)),
     DIKS_SMALL_F = ((DIKT_UNICODE) | (0x66)),
     DIKS_SMALL_G = ((DIKT_UNICODE) | (0x67)),
     DIKS_SMALL_H = ((DIKT_UNICODE) | (0x68)),
     DIKS_SMALL_I = ((DIKT_UNICODE) | (0x69)),
     DIKS_SMALL_J = ((DIKT_UNICODE) | (0x6A)),
     DIKS_SMALL_K = ((DIKT_UNICODE) | (0x6B)),
     DIKS_SMALL_L = ((DIKT_UNICODE) | (0x6C)),
     DIKS_SMALL_M = ((DIKT_UNICODE) | (0x6D)),
     DIKS_SMALL_N = ((DIKT_UNICODE) | (0x6E)),
     DIKS_SMALL_O = ((DIKT_UNICODE) | (0x6F)),
     DIKS_SMALL_P = ((DIKT_UNICODE) | (0x70)),
     DIKS_SMALL_Q = ((DIKT_UNICODE) | (0x71)),
     DIKS_SMALL_R = ((DIKT_UNICODE) | (0x72)),
     DIKS_SMALL_S = ((DIKT_UNICODE) | (0x73)),
     DIKS_SMALL_T = ((DIKT_UNICODE) | (0x74)),
     DIKS_SMALL_U = ((DIKT_UNICODE) | (0x75)),
     DIKS_SMALL_V = ((DIKT_UNICODE) | (0x76)),
     DIKS_SMALL_W = ((DIKT_UNICODE) | (0x77)),
     DIKS_SMALL_X = ((DIKT_UNICODE) | (0x78)),
     DIKS_SMALL_Y = ((DIKT_UNICODE) | (0x79)),
     DIKS_SMALL_Z = ((DIKT_UNICODE) | (0x7A)),
     DIKS_CURLY_BRACKET_LEFT = ((DIKT_UNICODE) | (0x7B)),
     DIKS_VERTICAL_BAR = ((DIKT_UNICODE) | (0x7C)),
     DIKS_CURLY_BRACKET_RIGHT = ((DIKT_UNICODE) | (0x7D)),
     DIKS_TILDE = ((DIKT_UNICODE) | (0x7E)),
     DIKS_DELETE = ((DIKT_UNICODE) | (0x7F)),

     DIKS_ENTER = DIKS_RETURN,




     DIKS_CURSOR_LEFT = ((DIKT_SPECIAL) | (0x00)),
     DIKS_CURSOR_RIGHT = ((DIKT_SPECIAL) | (0x01)),
     DIKS_CURSOR_UP = ((DIKT_SPECIAL) | (0x02)),
     DIKS_CURSOR_DOWN = ((DIKT_SPECIAL) | (0x03)),
     DIKS_INSERT = ((DIKT_SPECIAL) | (0x04)),
     DIKS_HOME = ((DIKT_SPECIAL) | (0x05)),
     DIKS_END = ((DIKT_SPECIAL) | (0x06)),
     DIKS_PAGE_UP = ((DIKT_SPECIAL) | (0x07)),
     DIKS_PAGE_DOWN = ((DIKT_SPECIAL) | (0x08)),
     DIKS_PRINT = ((DIKT_SPECIAL) | (0x09)),
     DIKS_PAUSE = ((DIKT_SPECIAL) | (0x0A)),
     DIKS_OK = ((DIKT_SPECIAL) | (0x0B)),
     DIKS_SELECT = ((DIKT_SPECIAL) | (0x0C)),
     DIKS_GOTO = ((DIKT_SPECIAL) | (0x0D)),
     DIKS_CLEAR = ((DIKT_SPECIAL) | (0x0E)),
     DIKS_POWER = ((DIKT_SPECIAL) | (0x0F)),
     DIKS_POWER2 = ((DIKT_SPECIAL) | (0x10)),
     DIKS_OPTION = ((DIKT_SPECIAL) | (0x11)),
     DIKS_MENU = ((DIKT_SPECIAL) | (0x12)),
     DIKS_HELP = ((DIKT_SPECIAL) | (0x13)),
     DIKS_INFO = ((DIKT_SPECIAL) | (0x14)),
     DIKS_TIME = ((DIKT_SPECIAL) | (0x15)),
     DIKS_VENDOR = ((DIKT_SPECIAL) | (0x16)),

     DIKS_ARCHIVE = ((DIKT_SPECIAL) | (0x17)),
     DIKS_PROGRAM = ((DIKT_SPECIAL) | (0x18)),
     DIKS_CHANNEL = ((DIKT_SPECIAL) | (0x19)),
     DIKS_FAVORITES = ((DIKT_SPECIAL) | (0x1A)),
     DIKS_EPG = ((DIKT_SPECIAL) | (0x1B)),
     DIKS_PVR = ((DIKT_SPECIAL) | (0x1C)),
     DIKS_MHP = ((DIKT_SPECIAL) | (0x1D)),
     DIKS_LANGUAGE = ((DIKT_SPECIAL) | (0x1E)),
     DIKS_TITLE = ((DIKT_SPECIAL) | (0x1F)),
     DIKS_SUBTITLE = ((DIKT_SPECIAL) | (0x20)),
     DIKS_ANGLE = ((DIKT_SPECIAL) | (0x21)),
     DIKS_ZOOM = ((DIKT_SPECIAL) | (0x22)),
     DIKS_MODE = ((DIKT_SPECIAL) | (0x23)),
     DIKS_KEYBOARD = ((DIKT_SPECIAL) | (0x24)),
     DIKS_PC = ((DIKT_SPECIAL) | (0x25)),
     DIKS_SCREEN = ((DIKT_SPECIAL) | (0x26)),

     DIKS_TV = ((DIKT_SPECIAL) | (0x27)),
     DIKS_TV2 = ((DIKT_SPECIAL) | (0x28)),
     DIKS_VCR = ((DIKT_SPECIAL) | (0x29)),
     DIKS_VCR2 = ((DIKT_SPECIAL) | (0x2A)),
     DIKS_SAT = ((DIKT_SPECIAL) | (0x2B)),
     DIKS_SAT2 = ((DIKT_SPECIAL) | (0x2C)),
     DIKS_CD = ((DIKT_SPECIAL) | (0x2D)),
     DIKS_TAPE = ((DIKT_SPECIAL) | (0x2E)),
     DIKS_RADIO = ((DIKT_SPECIAL) | (0x2F)),
     DIKS_TUNER = ((DIKT_SPECIAL) | (0x30)),
     DIKS_PLAYER = ((DIKT_SPECIAL) | (0x31)),
     DIKS_TEXT = ((DIKT_SPECIAL) | (0x32)),
     DIKS_DVD = ((DIKT_SPECIAL) | (0x33)),
     DIKS_AUX = ((DIKT_SPECIAL) | (0x34)),
     DIKS_MP3 = ((DIKT_SPECIAL) | (0x35)),
     DIKS_PHONE = ((DIKT_SPECIAL) | (0x36)),
     DIKS_AUDIO = ((DIKT_SPECIAL) | (0x37)),
     DIKS_VIDEO = ((DIKT_SPECIAL) | (0x38)),

     DIKS_INTERNET = ((DIKT_SPECIAL) | (0x39)),
     DIKS_MAIL = ((DIKT_SPECIAL) | (0x3A)),
     DIKS_NEWS = ((DIKT_SPECIAL) | (0x3B)),
     DIKS_DIRECTORY = ((DIKT_SPECIAL) | (0x3C)),
     DIKS_LIST = ((DIKT_SPECIAL) | (0x3D)),
     DIKS_CALCULATOR = ((DIKT_SPECIAL) | (0x3E)),
     DIKS_MEMO = ((DIKT_SPECIAL) | (0x3F)),
     DIKS_CALENDAR = ((DIKT_SPECIAL) | (0x40)),
     DIKS_EDITOR = ((DIKT_SPECIAL) | (0x41)),

     DIKS_RED = ((DIKT_SPECIAL) | (0x42)),
     DIKS_GREEN = ((DIKT_SPECIAL) | (0x43)),
     DIKS_YELLOW = ((DIKT_SPECIAL) | (0x44)),
     DIKS_BLUE = ((DIKT_SPECIAL) | (0x45)),

     DIKS_CHANNEL_UP = ((DIKT_SPECIAL) | (0x46)),
     DIKS_CHANNEL_DOWN = ((DIKT_SPECIAL) | (0x47)),
     DIKS_BACK = ((DIKT_SPECIAL) | (0x48)),
     DIKS_FORWARD = ((DIKT_SPECIAL) | (0x49)),
     DIKS_FIRST = ((DIKT_SPECIAL) | (0x4A)),
     DIKS_LAST = ((DIKT_SPECIAL) | (0x4B)),
     DIKS_VOLUME_UP = ((DIKT_SPECIAL) | (0x4C)),
     DIKS_VOLUME_DOWN = ((DIKT_SPECIAL) | (0x4D)),
     DIKS_MUTE = ((DIKT_SPECIAL) | (0x4E)),
     DIKS_AB = ((DIKT_SPECIAL) | (0x4F)),
     DIKS_PLAYPAUSE = ((DIKT_SPECIAL) | (0x50)),
     DIKS_PLAY = ((DIKT_SPECIAL) | (0x51)),
     DIKS_STOP = ((DIKT_SPECIAL) | (0x52)),
     DIKS_RESTART = ((DIKT_SPECIAL) | (0x53)),
     DIKS_SLOW = ((DIKT_SPECIAL) | (0x54)),
     DIKS_FAST = ((DIKT_SPECIAL) | (0x55)),
     DIKS_RECORD = ((DIKT_SPECIAL) | (0x56)),
     DIKS_EJECT = ((DIKT_SPECIAL) | (0x57)),
     DIKS_SHUFFLE = ((DIKT_SPECIAL) | (0x58)),
     DIKS_REWIND = ((DIKT_SPECIAL) | (0x59)),
     DIKS_FASTFORWARD = ((DIKT_SPECIAL) | (0x5A)),
     DIKS_PREVIOUS = ((DIKT_SPECIAL) | (0x5B)),
     DIKS_NEXT = ((DIKT_SPECIAL) | (0x5C)),
     DIKS_BEGIN = ((DIKT_SPECIAL) | (0x5D)),

     DIKS_DIGITS = ((DIKT_SPECIAL) | (0x5E)),
     DIKS_TEEN = ((DIKT_SPECIAL) | (0x5F)),
     DIKS_TWEN = ((DIKT_SPECIAL) | (0x60)),

     DIKS_BREAK = ((DIKT_SPECIAL) | (0x61)),
     DIKS_EXIT = ((DIKT_SPECIAL) | (0x62)),
     DIKS_SETUP = ((DIKT_SPECIAL) | (0x63)),

     DIKS_CURSOR_LEFT_UP = ((DIKT_SPECIAL) | (0x64)),
     DIKS_CURSOR_LEFT_DOWN = ((DIKT_SPECIAL) | (0x65)),
     DIKS_CURSOR_UP_RIGHT = ((DIKT_SPECIAL) | (0x66)),
     DIKS_CURSOR_DOWN_RIGHT = ((DIKT_SPECIAL) | (0x67)),






     DIKS_F1 = (((DIKT_FUNCTION) | (1))),
     DIKS_F2 = (((DIKT_FUNCTION) | (2))),
     DIKS_F3 = (((DIKT_FUNCTION) | (3))),
     DIKS_F4 = (((DIKT_FUNCTION) | (4))),
     DIKS_F5 = (((DIKT_FUNCTION) | (5))),
     DIKS_F6 = (((DIKT_FUNCTION) | (6))),
     DIKS_F7 = (((DIKT_FUNCTION) | (7))),
     DIKS_F8 = (((DIKT_FUNCTION) | (8))),
     DIKS_F9 = (((DIKT_FUNCTION) | (9))),
     DIKS_F10 = (((DIKT_FUNCTION) | (10))),
     DIKS_F11 = (((DIKT_FUNCTION) | (11))),
     DIKS_F12 = (((DIKT_FUNCTION) | (12))),




     DIKS_SHIFT = (((DIKT_MODIFIER) | ((1 << DIMKI_SHIFT)))),
     DIKS_CONTROL = (((DIKT_MODIFIER) | ((1 << DIMKI_CONTROL)))),
     DIKS_ALT = (((DIKT_MODIFIER) | ((1 << DIMKI_ALT)))),
     DIKS_ALTGR = (((DIKT_MODIFIER) | ((1 << DIMKI_ALTGR)))),
     DIKS_META = (((DIKT_MODIFIER) | ((1 << DIMKI_META)))),
     DIKS_SUPER = (((DIKT_MODIFIER) | ((1 << DIMKI_SUPER)))),
     DIKS_HYPER = (((DIKT_MODIFIER) | ((1 << DIMKI_HYPER)))),




     DIKS_CAPS_LOCK = ((DIKT_LOCK) | (0x00)),
     DIKS_NUM_LOCK = ((DIKT_LOCK) | (0x01)),
     DIKS_SCROLL_LOCK = ((DIKT_LOCK) | (0x02)),




     DIKS_DEAD_ABOVEDOT = ((DIKT_DEAD) | (0x00)),
     DIKS_DEAD_ABOVERING = ((DIKT_DEAD) | (0x01)),
     DIKS_DEAD_ACUTE = ((DIKT_DEAD) | (0x02)),
     DIKS_DEAD_BREVE = ((DIKT_DEAD) | (0x03)),
     DIKS_DEAD_CARON = ((DIKT_DEAD) | (0x04)),
     DIKS_DEAD_CEDILLA = ((DIKT_DEAD) | (0x05)),
     DIKS_DEAD_CIRCUMFLEX = ((DIKT_DEAD) | (0x06)),
     DIKS_DEAD_DIAERESIS = ((DIKT_DEAD) | (0x07)),
     DIKS_DEAD_DOUBLEACUTE = ((DIKT_DEAD) | (0x08)),
     DIKS_DEAD_GRAVE = ((DIKT_DEAD) | (0x09)),
     DIKS_DEAD_IOTA = ((DIKT_DEAD) | (0x0A)),
     DIKS_DEAD_MACRON = ((DIKT_DEAD) | (0x0B)),
     DIKS_DEAD_OGONEK = ((DIKT_DEAD) | (0x0C)),
     DIKS_DEAD_SEMIVOICED_SOUND = ((DIKT_DEAD) | (0x0D)),
     DIKS_DEAD_TILDE = ((DIKT_DEAD) | (0x0E)),
     DIKS_DEAD_VOICED_SOUND = ((DIKT_DEAD) | (0x0F)),






     DIKS_CUSTOM0 = (((DIKT_CUSTOM) | (0))),
     DIKS_CUSTOM1 = (((DIKT_CUSTOM) | (1))),
     DIKS_CUSTOM2 = (((DIKT_CUSTOM) | (2))),
     DIKS_CUSTOM3 = (((DIKT_CUSTOM) | (3))),
     DIKS_CUSTOM4 = (((DIKT_CUSTOM) | (4))),
     DIKS_CUSTOM5 = (((DIKT_CUSTOM) | (5))),
     DIKS_CUSTOM6 = (((DIKT_CUSTOM) | (6))),
     DIKS_CUSTOM7 = (((DIKT_CUSTOM) | (7))),
     DIKS_CUSTOM8 = (((DIKT_CUSTOM) | (8))),
     DIKS_CUSTOM9 = (((DIKT_CUSTOM) | (9))),
     DIKS_CUSTOM10 = (((DIKT_CUSTOM) | (10))),
     DIKS_CUSTOM11 = (((DIKT_CUSTOM) | (11))),
     DIKS_CUSTOM12 = (((DIKT_CUSTOM) | (12))),
     DIKS_CUSTOM13 = (((DIKT_CUSTOM) | (13))),
     DIKS_CUSTOM14 = (((DIKT_CUSTOM) | (14))),
     DIKS_CUSTOM15 = (((DIKT_CUSTOM) | (15))),
     DIKS_CUSTOM16 = (((DIKT_CUSTOM) | (16))),
     DIKS_CUSTOM17 = (((DIKT_CUSTOM) | (17))),
     DIKS_CUSTOM18 = (((DIKT_CUSTOM) | (18))),
     DIKS_CUSTOM19 = (((DIKT_CUSTOM) | (19))),
     DIKS_CUSTOM20 = (((DIKT_CUSTOM) | (20))),
     DIKS_CUSTOM21 = (((DIKT_CUSTOM) | (21))),
     DIKS_CUSTOM22 = (((DIKT_CUSTOM) | (22))),
     DIKS_CUSTOM23 = (((DIKT_CUSTOM) | (23))),
     DIKS_CUSTOM24 = (((DIKT_CUSTOM) | (24))),
     DIKS_CUSTOM25 = (((DIKT_CUSTOM) | (25))),
     DIKS_CUSTOM26 = (((DIKT_CUSTOM) | (26))),
     DIKS_CUSTOM27 = (((DIKT_CUSTOM) | (27))),
     DIKS_CUSTOM28 = (((DIKT_CUSTOM) | (28))),
     DIKS_CUSTOM29 = (((DIKT_CUSTOM) | (29))),
     DIKS_CUSTOM30 = (((DIKT_CUSTOM) | (30))),
     DIKS_CUSTOM31 = (((DIKT_CUSTOM) | (31))),
     DIKS_CUSTOM32 = (((DIKT_CUSTOM) | (32))),
     DIKS_CUSTOM33 = (((DIKT_CUSTOM) | (33))),
     DIKS_CUSTOM34 = (((DIKT_CUSTOM) | (34))),
     DIKS_CUSTOM35 = (((DIKT_CUSTOM) | (35))),
     DIKS_CUSTOM36 = (((DIKT_CUSTOM) | (36))),
     DIKS_CUSTOM37 = (((DIKT_CUSTOM) | (37))),
     DIKS_CUSTOM38 = (((DIKT_CUSTOM) | (38))),
     DIKS_CUSTOM39 = (((DIKT_CUSTOM) | (39))),
     DIKS_CUSTOM40 = (((DIKT_CUSTOM) | (40))),
     DIKS_CUSTOM41 = (((DIKT_CUSTOM) | (41))),
     DIKS_CUSTOM42 = (((DIKT_CUSTOM) | (42))),
     DIKS_CUSTOM43 = (((DIKT_CUSTOM) | (43))),
     DIKS_CUSTOM44 = (((DIKT_CUSTOM) | (44))),
     DIKS_CUSTOM45 = (((DIKT_CUSTOM) | (45))),
     DIKS_CUSTOM46 = (((DIKT_CUSTOM) | (46))),
     DIKS_CUSTOM47 = (((DIKT_CUSTOM) | (47))),
     DIKS_CUSTOM48 = (((DIKT_CUSTOM) | (48))),
     DIKS_CUSTOM49 = (((DIKT_CUSTOM) | (49))),
     DIKS_CUSTOM50 = (((DIKT_CUSTOM) | (50))),
     DIKS_CUSTOM51 = (((DIKT_CUSTOM) | (51))),
     DIKS_CUSTOM52 = (((DIKT_CUSTOM) | (52))),
     DIKS_CUSTOM53 = (((DIKT_CUSTOM) | (53))),
     DIKS_CUSTOM54 = (((DIKT_CUSTOM) | (54))),
     DIKS_CUSTOM55 = (((DIKT_CUSTOM) | (55))),
     DIKS_CUSTOM56 = (((DIKT_CUSTOM) | (56))),
     DIKS_CUSTOM57 = (((DIKT_CUSTOM) | (57))),
     DIKS_CUSTOM58 = (((DIKT_CUSTOM) | (58))),
     DIKS_CUSTOM59 = (((DIKT_CUSTOM) | (59))),
     DIKS_CUSTOM60 = (((DIKT_CUSTOM) | (60))),
     DIKS_CUSTOM61 = (((DIKT_CUSTOM) | (61))),
     DIKS_CUSTOM62 = (((DIKT_CUSTOM) | (62))),
     DIKS_CUSTOM63 = (((DIKT_CUSTOM) | (63))),
     DIKS_CUSTOM64 = (((DIKT_CUSTOM) | (64))),
     DIKS_CUSTOM65 = (((DIKT_CUSTOM) | (65))),
     DIKS_CUSTOM66 = (((DIKT_CUSTOM) | (66))),
     DIKS_CUSTOM67 = (((DIKT_CUSTOM) | (67))),
     DIKS_CUSTOM68 = (((DIKT_CUSTOM) | (68))),
     DIKS_CUSTOM69 = (((DIKT_CUSTOM) | (69))),
     DIKS_CUSTOM70 = (((DIKT_CUSTOM) | (70))),
     DIKS_CUSTOM71 = (((DIKT_CUSTOM) | (71))),
     DIKS_CUSTOM72 = (((DIKT_CUSTOM) | (72))),
     DIKS_CUSTOM73 = (((DIKT_CUSTOM) | (73))),
     DIKS_CUSTOM74 = (((DIKT_CUSTOM) | (74))),
     DIKS_CUSTOM75 = (((DIKT_CUSTOM) | (75))),
     DIKS_CUSTOM76 = (((DIKT_CUSTOM) | (76))),
     DIKS_CUSTOM77 = (((DIKT_CUSTOM) | (77))),
     DIKS_CUSTOM78 = (((DIKT_CUSTOM) | (78))),
     DIKS_CUSTOM79 = (((DIKT_CUSTOM) | (79))),
     DIKS_CUSTOM80 = (((DIKT_CUSTOM) | (80))),
     DIKS_CUSTOM81 = (((DIKT_CUSTOM) | (81))),
     DIKS_CUSTOM82 = (((DIKT_CUSTOM) | (82))),
     DIKS_CUSTOM83 = (((DIKT_CUSTOM) | (83))),
     DIKS_CUSTOM84 = (((DIKT_CUSTOM) | (84))),
     DIKS_CUSTOM85 = (((DIKT_CUSTOM) | (85))),
     DIKS_CUSTOM86 = (((DIKT_CUSTOM) | (86))),
     DIKS_CUSTOM87 = (((DIKT_CUSTOM) | (87))),
     DIKS_CUSTOM88 = (((DIKT_CUSTOM) | (88))),
     DIKS_CUSTOM89 = (((DIKT_CUSTOM) | (89))),
     DIKS_CUSTOM90 = (((DIKT_CUSTOM) | (90))),
     DIKS_CUSTOM91 = (((DIKT_CUSTOM) | (91))),
     DIKS_CUSTOM92 = (((DIKT_CUSTOM) | (92))),
     DIKS_CUSTOM93 = (((DIKT_CUSTOM) | (93))),
     DIKS_CUSTOM94 = (((DIKT_CUSTOM) | (94))),
     DIKS_CUSTOM95 = (((DIKT_CUSTOM) | (95))),
     DIKS_CUSTOM96 = (((DIKT_CUSTOM) | (96))),
     DIKS_CUSTOM97 = (((DIKT_CUSTOM) | (97))),
     DIKS_CUSTOM98 = (((DIKT_CUSTOM) | (98))),
     DIKS_CUSTOM99 = (((DIKT_CUSTOM) | (99)))
} DFBInputDeviceKeySymbol;




typedef enum {
     DILS_SCROLL = 0x00000001,
     DILS_NUM = 0x00000002,
     DILS_CAPS = 0x00000004
} DFBInputDeviceLockState;




typedef enum {
     DIKSI_BASE = 0x00,

     DIKSI_BASE_SHIFT = 0x01,

     DIKSI_ALT = 0x02,

     DIKSI_ALT_SHIFT = 0x03,


     DIKSI_LAST = DIKSI_ALT_SHIFT
} DFBInputDeviceKeymapSymbolIndex;




typedef struct {
     int code;

     DFBInputDeviceLockState locks;

     DFBInputDeviceKeyIdentifier identifier;
     DFBInputDeviceKeySymbol symbols[DIKSI_LAST+1];

} DFBInputDeviceKeymapEntry;
# 36 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h" 2
# 69 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
extern const unsigned int directfb_major_version;
extern const unsigned int directfb_minor_version;
extern const unsigned int directfb_micro_version;
extern const unsigned int directfb_binary_age;
extern const unsigned int directfb_interface_age;





const char * DirectFBCheckVersion( unsigned int required_major,
                                   unsigned int required_minor,
                                   unsigned int required_micro );





typedef struct _IDirectFB IDirectFB;




typedef struct _IDirectFBScreen IDirectFBScreen;




typedef struct _IDirectFBDisplayLayer IDirectFBDisplayLayer;





typedef struct _IDirectFBSurface IDirectFBSurface;




typedef struct _IDirectFBPalette IDirectFBPalette;





typedef struct _IDirectFBWindow IDirectFBWindow;




typedef struct _IDirectFBInputDevice IDirectFBInputDevice;




typedef struct _IDirectFBEventBuffer IDirectFBEventBuffer;




typedef struct _IDirectFBFont IDirectFBFont;




typedef struct _IDirectFBImageProvider IDirectFBImageProvider;




typedef struct _IDirectFBVideoProvider IDirectFBVideoProvider;




typedef struct _IDirectFBDataBuffer IDirectFBDataBuffer;




typedef struct _IDirectFBGL IDirectFBGL;







typedef enum {
     DFB_OK,
     DFB_FAILURE,
     DFB_INIT,
     DFB_BUG,
     DFB_DEAD,
     DFB_UNSUPPORTED,
     DFB_UNIMPLEMENTED,
     DFB_ACCESSDENIED,
     DFB_INVARG,
     DFB_NOSYSTEMMEMORY,
     DFB_NOVIDEOMEMORY,
     DFB_LOCKED,
     DFB_BUFFEREMPTY,
     DFB_FILENOTFOUND,
     DFB_IO,
     DFB_BUSY,
     DFB_NOIMPL,
     DFB_MISSINGFONT,
     DFB_TIMEOUT,
     DFB_MISSINGIMAGE,
     DFB_THIZNULL,
     DFB_IDNOTFOUND,
     DFB_INVAREA,
     DFB_DESTROYED,
     DFB_FUSION,
     DFB_BUFFERTOOLARGE,
     DFB_INTERRUPTED,
     DFB_NOCONTEXT,
     DFB_TEMPUNAVAIL,
     DFB_LIMITEXCEEDED,
     DFB_NOSUCHMETHOD,
     DFB_NOSUCHINSTANCE,
     DFB_ITEMNOTFOUND,
     DFB_VERSIONMISMATCH,
     DFB_NOSHAREDMEMORY,
     DFB_EOF,
     DFB_SUSPENDED,
     DFB_INCOMPLETE,
     DFB_NOCORE
} DFBResult;




typedef enum {
     DFB_FALSE = 0,
     DFB_TRUE = 1
} DFBBoolean;




typedef struct {
     int x;
     int y;
} DFBPoint;




typedef struct {
     int x;
     int w;
} DFBSpan;




typedef struct {
     int w;
     int h;
} DFBDimension;




typedef struct {
     int x;
     int y;
     int w;
     int h;
} DFBRectangle;






typedef struct {
     float x;
     float y;
     float w;
     float h;
} DFBLocation;






typedef struct {
     int x1;
     int y1;
     int x2;
     int y2;
} DFBRegion;






typedef struct {
     int l;
     int t;
     int r;
     int b;
} DFBInsets;




typedef struct {
     int x1;
     int y1;
     int x2;
     int y2;
     int x3;
     int y3;
} DFBTriangle;




typedef struct {
     u8 a;
     u8 r;
     u8 g;
     u8 b;
} DFBColor;




typedef struct {
     u8 index;
     u8 r;
     u8 g;
     u8 b;
} DFBColorKey;




typedef struct {
     u8 a;
     u8 y;
     u8 u;
     u8 v;
} DFBColorYUV;
# 363 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
DFBResult DirectFBError(
                             const char *msg,
                             DFBResult result
                       );




DFBResult DirectFBErrorFatal(
                             const char *msg,
                             DFBResult result
                            );




const char *DirectFBErrorString(
                         DFBResult result
                      );






const char *DirectFBUsageString( void );






DFBResult DirectFBInit(
                         int *argc,
                         char *(*argv[])
                      );






DFBResult DirectFBSetOption(
                         const char *name,
                         const char *value
                      );




DFBResult DirectFBCreate(
                          IDirectFB **interface

                        );


typedef unsigned int DFBScreenID;
typedef unsigned int DFBDisplayLayerID;
typedef unsigned int DFBDisplayLayerSourceID;
typedef unsigned int DFBWindowID;
typedef unsigned int DFBInputDeviceID;
typedef unsigned int DFBTextEncodingID;

typedef u32 DFBDisplayLayerIDs;
# 463 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DFSCL_NORMAL = 0x00000000,




     DFSCL_FULLSCREEN,



     DFSCL_EXCLUSIVE






} DFBCooperativeLevel;




typedef enum {
     DLCAPS_NONE = 0x00000000,

     DLCAPS_SURFACE = 0x00000001,



     DLCAPS_OPACITY = 0x00000002,

     DLCAPS_ALPHACHANNEL = 0x00000004,

     DLCAPS_SCREEN_LOCATION = 0x00000008,


     DLCAPS_FLICKER_FILTERING = 0x00000010,

     DLCAPS_DEINTERLACING = 0x00000020,


     DLCAPS_SRC_COLORKEY = 0x00000040,
     DLCAPS_DST_COLORKEY = 0x00000080,


     DLCAPS_BRIGHTNESS = 0x00000100,
     DLCAPS_CONTRAST = 0x00000200,
     DLCAPS_HUE = 0x00000400,
     DLCAPS_SATURATION = 0x00000800,
     DLCAPS_LEVELS = 0x00001000,

     DLCAPS_FIELD_PARITY = 0x00002000,
     DLCAPS_WINDOWS = 0x00004000,
     DLCAPS_SOURCES = 0x00008000,
     DLCAPS_ALPHA_RAMP = 0x00010000,





     DLCAPS_PREMULTIPLIED = 0x00020000,

     DLCAPS_SCREEN_POSITION = 0x00100000,
     DLCAPS_SCREEN_SIZE = 0x00200000,

     DLCAPS_CLIP_REGIONS = 0x00400000,

     DLCAPS_ALL = 0x0073FFFF
} DFBDisplayLayerCapabilities;




typedef enum {
     DSCCAPS_NONE = 0x00000000,

     DSCCAPS_VSYNC = 0x00000001,

     DSCCAPS_POWER_MANAGEMENT = 0x00000002,

     DSCCAPS_MIXERS = 0x00000010,
     DSCCAPS_ENCODERS = 0x00000020,
     DSCCAPS_OUTPUTS = 0x00000040,

     DSCCAPS_ALL = 0x00000073
} DFBScreenCapabilities;




typedef enum {
     DLOP_NONE = 0x00000000,
     DLOP_ALPHACHANNEL = 0x00000001,


     DLOP_FLICKER_FILTERING = 0x00000002,
     DLOP_DEINTERLACING = 0x00000004,

     DLOP_SRC_COLORKEY = 0x00000008,
     DLOP_DST_COLORKEY = 0x00000010,
     DLOP_OPACITY = 0x00000020,

     DLOP_FIELD_PARITY = 0x00000040
} DFBDisplayLayerOptions;




typedef enum {
     DLBM_UNKNOWN = 0x00000000,

     DLBM_FRONTONLY = 0x00000001,
     DLBM_BACKVIDEO = 0x00000002,
     DLBM_BACKSYSTEM = 0x00000004,
     DLBM_TRIPLE = 0x00000008,
     DLBM_WINDOWS = 0x00000010

} DFBDisplayLayerBufferMode;




typedef enum {
     DSDESC_NONE = 0x00000000,

     DSDESC_CAPS = 0x00000001,
     DSDESC_WIDTH = 0x00000002,
     DSDESC_HEIGHT = 0x00000004,
     DSDESC_PIXELFORMAT = 0x00000008,
     DSDESC_PREALLOCATED = 0x00000010,






     DSDESC_PALETTE = 0x00000020,



     DSDESC_RESOURCE_ID = 0x00000100,



     DSDESC_ALL = 0x0000013F
} DFBSurfaceDescriptionFlags;




typedef enum {
     DPDESC_CAPS = 0x00000001,
     DPDESC_SIZE = 0x00000002,
     DPDESC_ENTRIES = 0x00000004


} DFBPaletteDescriptionFlags;




typedef enum {
     DSCAPS_NONE = 0x00000000,

     DSCAPS_PRIMARY = 0x00000001,
     DSCAPS_SYSTEMONLY = 0x00000002,

     DSCAPS_VIDEOONLY = 0x00000004,

     DSCAPS_DOUBLE = 0x00000010,
     DSCAPS_SUBSURFACE = 0x00000020,

     DSCAPS_INTERLACED = 0x00000040,



     DSCAPS_SEPARATED = 0x00000080,



     DSCAPS_STATIC_ALLOC = 0x00000100,





     DSCAPS_TRIPLE = 0x00000200,

     DSCAPS_PREMULTIPLIED = 0x00001000,

     DSCAPS_DEPTH = 0x00010000,

     DSCAPS_SHARED = 0x00100000,

     DSCAPS_ALL = 0x001113F7,


     DSCAPS_FLIPPING = DSCAPS_DOUBLE | DSCAPS_TRIPLE

} DFBSurfaceCapabilities;




typedef enum {
     DPCAPS_NONE = 0x00000000
} DFBPaletteCapabilities;




typedef enum {
     DSDRAW_NOFX = 0x00000000,
     DSDRAW_BLEND = 0x00000001,
     DSDRAW_DST_COLORKEY = 0x00000002,

     DSDRAW_SRC_PREMULTIPLY = 0x00000004,

     DSDRAW_DST_PREMULTIPLY = 0x00000008,
     DSDRAW_DEMULTIPLY = 0x00000010,

     DSDRAW_XOR = 0x00000020

} DFBSurfaceDrawingFlags;




typedef enum {
     DSBLIT_NOFX = 0x00000000,
     DSBLIT_BLEND_ALPHACHANNEL = 0x00000001,

     DSBLIT_BLEND_COLORALPHA = 0x00000002,

     DSBLIT_COLORIZE = 0x00000004,

     DSBLIT_SRC_COLORKEY = 0x00000008,
     DSBLIT_DST_COLORKEY = 0x00000010,

     DSBLIT_SRC_PREMULTIPLY = 0x00000020,

     DSBLIT_DST_PREMULTIPLY = 0x00000040,
     DSBLIT_DEMULTIPLY = 0x00000080,

     DSBLIT_DEINTERLACE = 0x00000100,


     DSBLIT_SRC_PREMULTCOLOR = 0x00000200,
     DSBLIT_XOR = 0x00000400,

     DSBLIT_INDEX_TRANSLATION = 0x00000800,

     DSBLIT_ROTATE180 = 0x00001000,
     DSBLIT_COLORKEY_PROTECT = 0x00010000,
} DFBSurfaceBlittingFlags;




typedef enum {
     DSRO_NONE = 0x00000000,

     DSRO_SMOOTH_UPSCALE = 0x00000001,
     DSRO_SMOOTH_DOWNSCALE = 0x00000002,
     DSRO_MATRIX = 0x00000004,
     DSRO_ANTIALIAS = 0x00000008,

     DSRO_ALL = 0x0000000F
} DFBSurfaceRenderOptions;




typedef enum {
     DFXL_NONE = 0x00000000,

     DFXL_FILLRECTANGLE = 0x00000001,
     DFXL_DRAWRECTANGLE = 0x00000002,
     DFXL_DRAWLINE = 0x00000004,
     DFXL_FILLTRIANGLE = 0x00000008,

     DFXL_BLIT = 0x00010000,
     DFXL_STRETCHBLIT = 0x00020000,
     DFXL_TEXTRIANGLES = 0x00040000,

     DFXL_DRAWSTRING = 0x01000000,

     DFXL_ALL = 0x0107000F
} DFBAccelerationMask;
# 768 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DLTF_NONE = 0x00000000,

     DLTF_GRAPHICS = 0x00000001,
     DLTF_VIDEO = 0x00000002,
     DLTF_STILL_PICTURE = 0x00000004,
     DLTF_BACKGROUND = 0x00000008,

     DLTF_ALL = 0x0000000F
} DFBDisplayLayerTypeFlags;





typedef enum {
     DIDTF_NONE = 0x00000000,

     DIDTF_KEYBOARD = 0x00000001,
     DIDTF_MOUSE = 0x00000002,
     DIDTF_JOYSTICK = 0x00000004,
     DIDTF_REMOTE = 0x00000008,
     DIDTF_VIRTUAL = 0x00000010,

     DIDTF_ALL = 0x0000001F
} DFBInputDeviceTypeFlags;




typedef enum {
     DICAPS_KEYS = 0x00000001,
     DICAPS_AXES = 0x00000002,
     DICAPS_BUTTONS = 0x00000004,

     DICAPS_ALL = 0x00000007
} DFBInputDeviceCapabilities;




typedef enum {
     DIBI_LEFT = 0x00000000,
     DIBI_RIGHT = 0x00000001,
     DIBI_MIDDLE = 0x00000002,

     DIBI_FIRST = DIBI_LEFT,

     DIBI_LAST = 0x0000001F
} DFBInputDeviceButtonIdentifier;
# 826 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DIAI_X = 0x00000000,
     DIAI_Y = 0x00000001,
     DIAI_Z = 0x00000002,

     DIAI_FIRST = DIAI_X,

     DIAI_LAST = 0x0000001F
} DFBInputDeviceAxisIdentifier;




typedef enum {
     DWDESC_CAPS = 0x00000001,
     DWDESC_WIDTH = 0x00000002,
     DWDESC_HEIGHT = 0x00000004,
     DWDESC_PIXELFORMAT = 0x00000008,
     DWDESC_POSX = 0x00000010,
     DWDESC_POSY = 0x00000020,
     DWDESC_SURFACE_CAPS = 0x00000040,

     DWDESC_PARENT = 0x00000080,
     DWDESC_OPTIONS = 0x00000100,
     DWDESC_STACKING = 0x00000200,

     DWDESC_RESOURCE_ID = 0x00001000,
} DFBWindowDescriptionFlags;




typedef enum {
     DBDESC_FILE = 0x00000001,

     DBDESC_MEMORY = 0x00000002

} DFBDataBufferDescriptionFlags;




typedef enum {
     DWCAPS_NONE = 0x00000000,
     DWCAPS_ALPHACHANNEL = 0x00000001,

     DWCAPS_DOUBLEBUFFER = 0x00000002,







     DWCAPS_INPUTONLY = 0x00000004,


     DWCAPS_NODECORATION = 0x00000008,
     DWCAPS_ALL = 0x0000000F
} DFBWindowCapabilities;




typedef enum {
     DWOP_NONE = 0x00000000,
     DWOP_COLORKEYING = 0x00000001,
     DWOP_ALPHACHANNEL = 0x00000002,

     DWOP_OPAQUE_REGION = 0x00000004,

     DWOP_SHAPED = 0x00000008,


     DWOP_KEEP_POSITION = 0x00000010,

     DWOP_KEEP_SIZE = 0x00000020,

     DWOP_KEEP_STACKING = 0x00000040,

     DWOP_GHOST = 0x00001000,


     DWOP_INDESTRUCTIBLE = 0x00002000,

     DWOP_SCALE = 0x00010000,


     DWOP_KEEP_ABOVE = 0x00100000,
     DWOP_KEEP_UNDER = 0x00200000,

     DWOP_ALL = 0x0031307F
} DFBWindowOptions;




typedef enum {
     DWSC_MIDDLE = 0x00000000,

     DWSC_UPPER = 0x00000001,




     DWSC_LOWER = 0x00000002




} DFBWindowStackingClass;
# 948 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DFFA_NONE = 0x00000000,
     DFFA_NOKERNING = 0x00000001,
     DFFA_NOHINTING = 0x00000002,
     DFFA_MONOCHROME = 0x00000004,
     DFFA_NOCHARMAP = 0x00000008,

     DFFA_FIXEDCLIP = 0x00000010
} DFBFontAttributes;




typedef enum {
     DFDESC_ATTRIBUTES = 0x00000001,
     DFDESC_HEIGHT = 0x00000002,
     DFDESC_WIDTH = 0x00000004,
     DFDESC_INDEX = 0x00000008,
     DFDESC_FIXEDADVANCE = 0x00000010,


     DFDESC_FRACT_HEIGHT = 0x00000020,
     DFDESC_FRACT_WIDTH = 0x00000040,
} DFBFontDescriptionFlags;
# 989 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef struct {
     DFBFontDescriptionFlags flags;

     DFBFontAttributes attributes;
     int height;
     int width;
     unsigned int index;
     int fixed_advance;

     int fract_height;
     int fract_width;
} DFBFontDescription;
# 1039 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DSPF_UNKNOWN = 0x00000000,


     DSPF_ARGB1555 = ( (((0 ) & 0x7F) ) | (((15) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB16 = ( (((1 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB24 = ( (((2 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB32 = ( (((3 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB = ( (((4 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_A8 = ( (((5 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_YUY2 = ( (((6 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB332 = ( (((7 ) & 0x7F) ) | (((8) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_UYVY = ( (((8 ) & 0x7F) ) | (((16) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_I420 = ( (((9 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_YV12 = ( (((10 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_LUT8 = ( (((11 ) & 0x7F) ) | (((8) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ALUT44 = ( (((12 ) & 0x7F) ) | (((4) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_AiRGB = ( (((13 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((1 ) ? 1 :0) << 31) ),


     DSPF_A1 = ( (((14 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((1 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((7 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV12 = ( (((15 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV16 = ( (((16 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((1 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB2554 = ( (((17 ) & 0x7F) ) | (((14) & 0x1F) << 7) | (((2) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB4444 = ( (((18 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_NV21 = ( (((19 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((1 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((2 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_AYUV = ( (((20 ) & 0x7F) ) | (((24) & 0x1F) << 7) | (((8) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((4 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_A4 = ( (((21 ) & 0x7F) ) | (((0) & 0x1F) << 7) | (((4) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((4 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((1 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB1666 = ( (((22 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((1) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_ARGB6666 = ( (((23 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((6) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB18 = ( (((24 ) & 0x7F) ) | (((18) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((3 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_LUT2 = ( (((25 ) & 0x7F) ) | (((2) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((1 ) ? 1 :0) << 16) | (((2 ) & 0x07) << 17) | (((0 ) & 0x07) << 20) | (((3 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((1 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB444 = ( (((26 ) & 0x7F) ) | (((12) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) ),


     DSPF_RGB555 = ( (((27 ) & 0x7F) ) | (((15) & 0x1F) << 7) | (((0) & 0x0F) << 12) | (((0 ) ? 1 :0) << 16) | (((0 ) & 0x07) << 17) | (((2 ) & 0x07) << 20) | (((0 ) & 0x07) << 23) | (((0 ) & 0x03) << 26) | (((0 ) & 0x03) << 28) | (((0 ) ? 1 :0) << 30) | (((0 ) ? 1 :0) << 31) )

} DFBSurfacePixelFormat;
# 1160 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef struct {
     DFBSurfaceDescriptionFlags flags;

     DFBSurfaceCapabilities caps;
     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;

     struct {
          void *data;
          int pitch;
     } preallocated[2];

     struct {
          const DFBColor *entries;
          unsigned int size;
     } palette;

     unsigned long resource_id;

} DFBSurfaceDescription;




typedef struct {
     DFBPaletteDescriptionFlags flags;

     DFBPaletteCapabilities caps;
     unsigned int size;
     const DFBColor *entries;

} DFBPaletteDescription;







typedef struct {
     DFBDisplayLayerTypeFlags type;
     DFBDisplayLayerCapabilities caps;

     char name[32];

     int level;
     int regions;



     int sources;
     int clip_regions;
} DFBDisplayLayerDescription;




typedef enum {
     DDLSCAPS_NONE = 0x00000000,

     DDLSCAPS_SURFACE = 0x00000001,

     DDLSCAPS_ALL = 0x00000001
} DFBDisplayLayerSourceCaps;






typedef struct {
     DFBDisplayLayerSourceID source_id;

     char name[24];

     DFBDisplayLayerSourceCaps caps;
} DFBDisplayLayerSourceDescription;







typedef struct {
     DFBScreenCapabilities caps;


     char name[32];

     int mixers;

     int encoders;

     int outputs;

} DFBScreenDescription;
# 1266 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef struct {
     DFBInputDeviceTypeFlags type;

     DFBInputDeviceCapabilities caps;



     int min_keycode;




     int max_keycode;




     DFBInputDeviceAxisIdentifier max_axis;

     DFBInputDeviceButtonIdentifier max_button;


     char name[32];

     char vendor[40];
} DFBInputDeviceDescription;





typedef struct {
     int major;
     int minor;

     char name[40];
     char vendor[60];
} DFBGraphicsDriverInfo;







typedef struct {
     DFBAccelerationMask acceleration_mask;

     DFBSurfaceBlittingFlags blitting_flags;
     DFBSurfaceDrawingFlags drawing_flags;

     unsigned int video_memory;

     char name[48];
     char vendor[64];

     DFBGraphicsDriverInfo driver;
} DFBGraphicsDeviceDescription;




typedef struct {
     DFBWindowDescriptionFlags flags;

     DFBWindowCapabilities caps;
     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;
     int posx;
     int posy;
     DFBSurfaceCapabilities surface_caps;
     DFBWindowID parent_id;
     DFBWindowOptions options;
     DFBWindowStackingClass stacking;

     unsigned long resource_id;
} DFBWindowDescription;




typedef struct {
     DFBDataBufferDescriptionFlags flags;

     const char *file;

     struct {
          const void *data;
          unsigned int length;
     } memory;
} DFBDataBufferDescription;




typedef enum {
     DFENUM_OK = 0x00000000,
     DFENUM_CANCEL = 0x00000001
} DFBEnumerationResult;




typedef DFBEnumerationResult (*DFBVideoModeCallback) (
     int width,
     int height,
     int bpp,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBScreenCallback) (
     DFBScreenID screen_id,
     DFBScreenDescription desc,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBDisplayLayerCallback) (
     DFBDisplayLayerID layer_id,
     DFBDisplayLayerDescription desc,
     void *callbackdata
);





typedef DFBEnumerationResult (*DFBInputDeviceCallback) (
     DFBInputDeviceID device_id,
     DFBInputDeviceDescription desc,
     void *callbackdata
);







typedef int (*DFBGetDataCallback) (
     void *buffer,
     unsigned int length,
     void *callbackdata
);




typedef enum {
     DVCAPS_BASIC = 0x00000000,
     DVCAPS_SEEK = 0x00000001,
     DVCAPS_SCALE = 0x00000002,
     DVCAPS_INTERLACED = 0x00000004,
     DVCAPS_SPEED = 0x00000008,
     DVCAPS_BRIGHTNESS = 0x00000010,
     DVCAPS_CONTRAST = 0x00000020,
     DVCAPS_HUE = 0x00000040,
     DVCAPS_SATURATION = 0x00000080,
     DVCAPS_INTERACTIVE = 0x00000100,
     DVCAPS_VOLUME = 0x00000200,
     DVCAPS_EVENT = 0x00000400,
     DVCAPS_ATTRIBUTES = 0x00000800,
     DVCAPS_AUDIO_SEL = 0x00001000,
} DFBVideoProviderCapabilities;




typedef enum {
     DVSTATE_UNKNOWN = 0x00000000,
     DVSTATE_PLAY = 0x00000001,
     DVSTATE_STOP = 0x00000002,
     DVSTATE_FINISHED = 0x00000003,
     DVSTATE_BUFFERING = 0x00000004

} DFBVideoProviderStatus;




typedef enum {
     DVPLAY_NOFX = 0x00000000,
     DVPLAY_REWIND = 0x00000001,
     DVPLAY_LOOPING = 0x00000002


} DFBVideoProviderPlaybackFlags;




typedef enum {
     DVAUDIOUNIT_NONE = 0x00000000,
     DVAUDIOUNIT_ONE = 0x00000001,
     DVAUDIOUNIT_TWO = 0x00000002,
     DVAUDIOUNIT_THREE = 0x00000004,
     DVAUDIOUNIT_FOUR = 0x00000008,
     DVAUDIOUNIT_ALL = 0x0000000F,
} DFBVideoProviderAudioUnits;





typedef enum {
     DCAF_NONE = 0x00000000,
     DCAF_BRIGHTNESS = 0x00000001,
     DCAF_CONTRAST = 0x00000002,
     DCAF_HUE = 0x00000004,
     DCAF_SATURATION = 0x00000008,
     DCAF_ALL = 0x0000000F
} DFBColorAdjustmentFlags;







typedef struct {
     DFBColorAdjustmentFlags flags;

     u16 brightness;
     u16 contrast;
     u16 hue;
     u16 saturation;
} DFBColorAdjustment;
# 1561 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFB { void *priv; int magic; DFBResult (*AddRef)( IDirectFB *thiz ); DFBResult (*Release)( IDirectFB *thiz ); DFBResult (*SetCooperativeLevel) ( IDirectFB *thiz, DFBCooperativeLevel level ); DFBResult (*SetVideoMode) ( IDirectFB *thiz, int width, int height, int bpp ); DFBResult (*GetDeviceDescription) ( IDirectFB *thiz, DFBGraphicsDeviceDescription *ret_desc ); DFBResult (*EnumVideoModes) ( IDirectFB *thiz, DFBVideoModeCallback callback, void *callbackdata ); DFBResult (*CreateSurface) ( IDirectFB *thiz, const DFBSurfaceDescription *desc, IDirectFBSurface **ret_interface ); DFBResult (*CreatePalette) ( IDirectFB *thiz, const DFBPaletteDescription *desc, IDirectFBPalette **ret_interface ); DFBResult (*EnumScreens) ( IDirectFB *thiz, DFBScreenCallback callback, void *callbackdata ); DFBResult (*GetScreen) ( IDirectFB *thiz, DFBScreenID screen_id, IDirectFBScreen **ret_interface ); DFBResult (*EnumDisplayLayers) ( IDirectFB *thiz, DFBDisplayLayerCallback callback, void *callbackdata ); DFBResult (*GetDisplayLayer) ( IDirectFB *thiz, DFBDisplayLayerID layer_id, IDirectFBDisplayLayer **ret_interface ); DFBResult (*EnumInputDevices) ( IDirectFB *thiz, DFBInputDeviceCallback callback, void *callbackdata ); DFBResult (*GetInputDevice) ( IDirectFB *thiz, DFBInputDeviceID device_id, IDirectFBInputDevice **ret_interface ); DFBResult (*CreateEventBuffer) ( IDirectFB *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*CreateInputEventBuffer) ( IDirectFB *thiz, DFBInputDeviceCapabilities caps, DFBBoolean global, IDirectFBEventBuffer **ret_buffer ); DFBResult (*CreateImageProvider) ( IDirectFB *thiz, const char *filename, IDirectFBImageProvider **ret_interface ); DFBResult (*CreateVideoProvider) ( IDirectFB *thiz, const char *filename, IDirectFBVideoProvider **ret_interface ); DFBResult (*CreateFont) ( IDirectFB *thiz, const char *filename, const DFBFontDescription *desc, IDirectFBFont **ret_interface ); DFBResult (*CreateDataBuffer) ( IDirectFB *thiz, const DFBDataBufferDescription *desc, IDirectFBDataBuffer **ret_interface ); DFBResult (*SetClipboardData) ( IDirectFB *thiz, const char *mime_type, const void *data, unsigned int size, struct timeval *ret_timestamp ); DFBResult (*GetClipboardData) ( IDirectFB *thiz, char **ret_mimetype, void **ret_data, unsigned int *ret_size ); DFBResult (*GetClipboardTimeStamp) ( IDirectFB *thiz, struct timeval *ret_timestamp ); DFBResult (*Suspend) ( IDirectFB *thiz ); DFBResult (*Resume) ( IDirectFB *thiz ); DFBResult (*WaitIdle) ( IDirectFB *thiz ); DFBResult (*WaitForSync) ( IDirectFB *thiz ); DFBResult (*GetInterface) ( IDirectFB *thiz, const char *type, const char *implementation, void *arg, void **ret_interface ); };
# 1924 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DLSCL_SHARED = 0,
     DLSCL_EXCLUSIVE,

     DLSCL_ADMINISTRATIVE

} DFBDisplayLayerCooperativeLevel;





typedef enum {
     DLBM_DONTCARE = 0,

     DLBM_COLOR,

     DLBM_IMAGE,
     DLBM_TILE
} DFBDisplayLayerBackgroundMode;




typedef enum {
     DLCONF_NONE = 0x00000000,

     DLCONF_WIDTH = 0x00000001,
     DLCONF_HEIGHT = 0x00000002,
     DLCONF_PIXELFORMAT = 0x00000004,
     DLCONF_BUFFERMODE = 0x00000008,
     DLCONF_OPTIONS = 0x00000010,
     DLCONF_SOURCE = 0x00000020,
     DLCONF_SURFACE_CAPS = 0x00000040,

     DLCONF_ALL = 0x0000007F
} DFBDisplayLayerConfigFlags;




typedef struct {
     DFBDisplayLayerConfigFlags flags;

     int width;
     int height;
     DFBSurfacePixelFormat pixelformat;
     DFBDisplayLayerBufferMode buffermode;
     DFBDisplayLayerOptions options;
     DFBDisplayLayerSourceID source;

     DFBSurfaceCapabilities surface_caps;

} DFBDisplayLayerConfig;




typedef enum {
     DSPM_ON = 0,
     DSPM_STANDBY,
     DSPM_SUSPEND,
     DSPM_OFF
} DFBScreenPowerMode;





typedef enum {
     DSMCAPS_NONE = 0x00000000,

     DSMCAPS_FULL = 0x00000001,
     DSMCAPS_SUB_LEVEL = 0x00000002,
     DSMCAPS_SUB_LAYERS = 0x00000004,
     DSMCAPS_BACKGROUND = 0x00000008
} DFBScreenMixerCapabilities;







typedef struct {
     DFBScreenMixerCapabilities caps;

     DFBDisplayLayerIDs layers;


     int sub_num;

     DFBDisplayLayerIDs sub_layers;


     char name[24];
} DFBScreenMixerDescription;




typedef enum {
     DSMCONF_NONE = 0x00000000,

     DSMCONF_TREE = 0x00000001,
     DSMCONF_LEVEL = 0x00000002,
     DSMCONF_LAYERS = 0x00000004,

     DSMCONF_BACKGROUND = 0x00000010,

     DSMCONF_ALL = 0x00000017
} DFBScreenMixerConfigFlags;




typedef enum {
     DSMT_UNKNOWN = 0x00000000,

     DSMT_FULL = 0x00000001,
     DSMT_SUB_LEVEL = 0x00000002,
     DSMT_SUB_LAYERS = 0x00000003
} DFBScreenMixerTree;




typedef struct {
     DFBScreenMixerConfigFlags flags;

     DFBScreenMixerTree tree;

     int level;
     DFBDisplayLayerIDs layers;

     DFBColor background;
} DFBScreenMixerConfig;





typedef enum {
     DSOCAPS_NONE = 0x00000000,

     DSOCAPS_CONNECTORS = 0x00000001,

     DSOCAPS_ENCODER_SEL = 0x00000010,
     DSOCAPS_SIGNAL_SEL = 0x00000020,
     DSOCAPS_CONNECTOR_SEL = 0x00000040,
     DSOCAPS_SLOW_BLANKING = 0x00000080,
     DSOCAPS_RESOLUTION = 0x00000100,
     DSOCAPS_ALL = 0x000001F1
} DFBScreenOutputCapabilities;




typedef enum {
     DSOC_UNKNOWN = 0x00000000,

     DSOC_VGA = 0x00000001,
     DSOC_SCART = 0x00000002,
     DSOC_YC = 0x00000004,
     DSOC_CVBS = 0x00000008,
     DSOC_SCART2 = 0x00000010,
     DSOC_COMPONENT = 0x00000020,
     DSOC_HDMI = 0x00000040
} DFBScreenOutputConnectors;




typedef enum {
     DSOS_NONE = 0x00000000,

     DSOS_VGA = 0x00000001,
     DSOS_YC = 0x00000002,
     DSOS_CVBS = 0x00000004,
     DSOS_RGB = 0x00000008,
     DSOS_YCBCR = 0x00000010,
     DSOS_HDMI = 0x00000020,
     DSOS_656 = 0x00000040
} DFBScreenOutputSignals;





typedef enum {
     DSOSB_OFF = 0x00000000,
     DSOSB_16x9 = 0x00000001,
     DSOSB_4x3 = 0x00000002,
     DSOSB_FOLLOW = 0x00000004,
     DSOSB_MONITOR = 0x00000008
} DFBScreenOutputSlowBlankingSignals;





typedef enum {
    DSOR_UNKNOWN = 0x00000000,
    DSOR_640_480 = 0x00000001,
    DSOR_720_480 = 0x00000002,
    DSOR_720_576 = 0x00000004,
    DSOR_800_600 = 0x00000008,
    DSOR_1024_768 = 0x00000010,
    DSOR_1152_864 = 0x00000020,
    DSOR_1280_720 = 0x00000040,
    DSOR_1280_768 = 0x00000080,
    DSOR_1280_960 = 0x00000100,
    DSOR_1280_1024 = 0x00000200,
    DSOR_1400_1050 = 0x00000400,
    DSOR_1600_1200 = 0x00000800,
    DSOR_1920_1080 = 0x00001000,
    DSOR_ALL = 0x00001FFF
} DFBScreenOutputResolution;







typedef struct {
     DFBScreenOutputCapabilities caps;

     DFBScreenOutputConnectors all_connectors;
     DFBScreenOutputSignals all_signals;
     DFBScreenOutputResolution all_resolutions;

     char name[24];
} DFBScreenOutputDescription;




typedef enum {
     DSOCONF_NONE = 0x00000000,

     DSOCONF_ENCODER = 0x00000001,
     DSOCONF_SIGNALS = 0x00000002,
     DSOCONF_CONNECTORS = 0x00000004,
     DSOCONF_SLOW_BLANKING= 0x00000008,
     DSOCONF_RESOLUTION = 0x00000010,

     DSOCONF_ALL = 0x0000001F
} DFBScreenOutputConfigFlags;




typedef struct {
     DFBScreenOutputConfigFlags flags;

     int encoder;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors out_connectors;
     DFBScreenOutputSlowBlankingSignals slow_blanking;
     DFBScreenOutputResolution resolution;
} DFBScreenOutputConfig;





typedef enum {
     DSECAPS_NONE = 0x00000000,

     DSECAPS_TV_STANDARDS = 0x00000001,
     DSECAPS_TEST_PICTURE = 0x00000002,
     DSECAPS_MIXER_SEL = 0x00000004,
     DSECAPS_OUT_SIGNALS = 0x00000008,
     DSECAPS_SCANMODE = 0x00000010,
     DSECAPS_FREQUENCY = 0x00000020,

     DSECAPS_BRIGHTNESS = 0x00000100,
     DSECAPS_CONTRAST = 0x00000200,
     DSECAPS_HUE = 0x00000400,
     DSECAPS_SATURATION = 0x00000800,

     DSECAPS_CONNECTORS = 0x00001000,
     DSECAPS_SLOW_BLANKING = 0x00002000,
     DSECAPS_RESOLUTION = 0x00004000,

     DSECAPS_ALL = 0x00007f3f
} DFBScreenEncoderCapabilities;




typedef enum {
     DSET_UNKNOWN = 0x00000000,

     DSET_CRTC = 0x00000001,
     DSET_TV = 0x00000002,
     DSET_DIGITAL = 0x00000004
} DFBScreenEncoderType;




typedef enum {
     DSETV_UNKNOWN = 0x00000000,

     DSETV_PAL = 0x00000001,
     DSETV_NTSC = 0x00000002,
     DSETV_SECAM = 0x00000004,
     DSETV_PAL_60 = 0x00000008,
     DSETV_PAL_BG = 0x00000010,
     DSETV_PAL_I = 0x00000020,
     DSETV_PAL_M = 0x00000040,
     DSETV_PAL_N = 0x00000080,
     DSETV_PAL_NC = 0x00000100,
     DSETV_NTSC_M_JPN = 0x00000200,
     DSETV_DIGITAL = 0x00000400,
     DSETV_ALL = 0x000007FF
} DFBScreenEncoderTVStandards;




typedef enum {
     DSESM_UNKNOWN = 0x00000000,

     DSESM_INTERLACED = 0x00000001,
     DSESM_PROGRESSIVE = 0x00000002
} DFBScreenEncoderScanMode;




typedef enum {
     DSEF_UNKNOWN = 0x00000000,

     DSEF_25HZ = 0x00000001,
     DSEF_29_97HZ = 0x00000002,
     DSEF_50HZ = 0x00000004,
     DSEF_59_94HZ = 0x00000008,
     DSEF_60HZ = 0x00000010,
     DSEF_75HZ = 0x00000020,
} DFBScreenEncoderFrequency;






typedef struct {
     DFBScreenEncoderCapabilities caps;
     DFBScreenEncoderType type;

     DFBScreenEncoderTVStandards tv_standards;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors all_connectors;
     DFBScreenOutputResolution all_resolutions;

     char name[24];
} DFBScreenEncoderDescription;




typedef enum {
     DSECONF_NONE = 0x00000000,

     DSECONF_TV_STANDARD = 0x00000001,
     DSECONF_TEST_PICTURE = 0x00000002,
     DSECONF_MIXER = 0x00000004,
     DSECONF_OUT_SIGNALS = 0x00000008,
     DSECONF_SCANMODE = 0x00000010,
     DSECONF_TEST_COLOR = 0x00000020,
     DSECONF_ADJUSTMENT = 0x00000040,
     DSECONF_FREQUENCY = 0x00000080,

     DSECONF_CONNECTORS = 0x00000100,
     DSECONF_SLOW_BLANKING = 0x00000200,
     DSECONF_RESOLUTION = 0x00000400,

     DSECONF_ALL = 0x000007FF
} DFBScreenEncoderConfigFlags;




typedef enum {
     DSETP_OFF = 0x00000000,

     DSETP_MULTI = 0x00000001,
     DSETP_SINGLE = 0x00000002,

     DSETP_WHITE = 0x00000010,
     DSETP_YELLOW = 0x00000020,
     DSETP_CYAN = 0x00000030,
     DSETP_GREEN = 0x00000040,
     DSETP_MAGENTA = 0x00000050,
     DSETP_RED = 0x00000060,
     DSETP_BLUE = 0x00000070,
     DSETP_BLACK = 0x00000080
} DFBScreenEncoderTestPicture;




typedef struct {
     DFBScreenEncoderConfigFlags flags;

     DFBScreenEncoderTVStandards tv_standard;
     DFBScreenEncoderTestPicture test_picture;
     int mixer;
     DFBScreenOutputSignals out_signals;
     DFBScreenOutputConnectors out_connectors;
     DFBScreenOutputSlowBlankingSignals slow_blanking;

     DFBScreenEncoderScanMode scanmode;

     DFBColor test_color;

     DFBColorAdjustment adjustment;

     DFBScreenEncoderFrequency frequency;
     DFBScreenOutputResolution resolution;
} DFBScreenEncoderConfig;
# 2357 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBScreen { void *priv; int magic; DFBResult (*AddRef)( IDirectFBScreen *thiz ); DFBResult (*Release)( IDirectFBScreen *thiz ); DFBResult (*GetID) ( IDirectFBScreen *thiz, DFBScreenID *ret_screen_id ); DFBResult (*GetDescription) ( IDirectFBScreen *thiz, DFBScreenDescription *ret_desc ); DFBResult (*GetSize) ( IDirectFBScreen *thiz, int *ret_width, int *ret_height ); DFBResult (*EnumDisplayLayers) ( IDirectFBScreen *thiz, DFBDisplayLayerCallback callback, void *callbackdata ); DFBResult (*SetPowerMode) ( IDirectFBScreen *thiz, DFBScreenPowerMode mode ); DFBResult (*WaitForSync) ( IDirectFBScreen *thiz ); DFBResult (*GetMixerDescriptions) ( IDirectFBScreen *thiz, DFBScreenMixerDescription *ret_descriptions ); DFBResult (*GetMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, DFBScreenMixerConfig *ret_config ); DFBResult (*TestMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, const DFBScreenMixerConfig *config, DFBScreenMixerConfigFlags *ret_failed ); DFBResult (*SetMixerConfiguration) ( IDirectFBScreen *thiz, int mixer, const DFBScreenMixerConfig *config ); DFBResult (*GetEncoderDescriptions) ( IDirectFBScreen *thiz, DFBScreenEncoderDescription *ret_descriptions ); DFBResult (*GetEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, DFBScreenEncoderConfig *ret_config ); DFBResult (*TestEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, const DFBScreenEncoderConfig *config, DFBScreenEncoderConfigFlags *ret_failed ); DFBResult (*SetEncoderConfiguration) ( IDirectFBScreen *thiz, int encoder, const DFBScreenEncoderConfig *config ); DFBResult (*GetOutputDescriptions) ( IDirectFBScreen *thiz, DFBScreenOutputDescription *ret_descriptions ); DFBResult (*GetOutputConfiguration) ( IDirectFBScreen *thiz, int output, DFBScreenOutputConfig *ret_config ); DFBResult (*TestOutputConfiguration) ( IDirectFBScreen *thiz, int output, const DFBScreenOutputConfig *config, DFBScreenOutputConfigFlags *ret_failed ); DFBResult (*SetOutputConfiguration) ( IDirectFBScreen *thiz, int output, const DFBScreenOutputConfig *config ); };
# 2572 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBDisplayLayer { void *priv; int magic; DFBResult (*AddRef)( IDirectFBDisplayLayer *thiz ); DFBResult (*Release)( IDirectFBDisplayLayer *thiz ); DFBResult (*GetID) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerID *ret_layer_id ); DFBResult (*GetDescription) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerDescription *ret_desc ); DFBResult (*GetSourceDescriptions) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerSourceDescription *ret_descriptions ); DFBResult (*GetCurrentOutputField) ( IDirectFBDisplayLayer *thiz, int *ret_field ); DFBResult (*GetSurface) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface **ret_interface ); DFBResult (*GetScreen) ( IDirectFBDisplayLayer *thiz, IDirectFBScreen **ret_interface ); DFBResult (*SetCooperativeLevel) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerCooperativeLevel level ); DFBResult (*GetConfiguration) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerConfig *ret_config ); DFBResult (*TestConfiguration) ( IDirectFBDisplayLayer *thiz, const DFBDisplayLayerConfig *config, DFBDisplayLayerConfigFlags *ret_failed ); DFBResult (*SetConfiguration) ( IDirectFBDisplayLayer *thiz, const DFBDisplayLayerConfig *config ); DFBResult (*SetScreenLocation) ( IDirectFBDisplayLayer *thiz, float x, float y, float width, float height ); DFBResult (*SetScreenPosition) ( IDirectFBDisplayLayer *thiz, int x, int y ); DFBResult (*SetScreenRectangle) ( IDirectFBDisplayLayer *thiz, int x, int y, int width, int height ); DFBResult (*SetOpacity) ( IDirectFBDisplayLayer *thiz, u8 opacity ); DFBResult (*SetSourceRectangle) ( IDirectFBDisplayLayer *thiz, int x, int y, int width, int height ); DFBResult (*SetFieldParity) ( IDirectFBDisplayLayer *thiz, int field ); DFBResult (*SetClipRegions) ( IDirectFBDisplayLayer *thiz, const DFBRegion *regions, int num_regions, DFBBoolean positive ); DFBResult (*SetSrcColorKey) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetDstColorKey) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b ); DFBResult (*GetLevel) ( IDirectFBDisplayLayer *thiz, int *ret_level ); DFBResult (*SetLevel) ( IDirectFBDisplayLayer *thiz, int level ); DFBResult (*SetBackgroundMode) ( IDirectFBDisplayLayer *thiz, DFBDisplayLayerBackgroundMode mode ); DFBResult (*SetBackgroundImage) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface *surface ); DFBResult (*SetBackgroundColor) ( IDirectFBDisplayLayer *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*GetColorAdjustment) ( IDirectFBDisplayLayer *thiz, DFBColorAdjustment *ret_adj ); DFBResult (*SetColorAdjustment) ( IDirectFBDisplayLayer *thiz, const DFBColorAdjustment *adj ); DFBResult (*CreateWindow) ( IDirectFBDisplayLayer *thiz, const DFBWindowDescription *desc, IDirectFBWindow **ret_interface ); DFBResult (*GetWindow) ( IDirectFBDisplayLayer *thiz, DFBWindowID window_id, IDirectFBWindow **ret_interface ); DFBResult (*EnableCursor) ( IDirectFBDisplayLayer *thiz, int enable ); DFBResult (*GetCursorPosition) ( IDirectFBDisplayLayer *thiz, int *ret_x, int *ret_y ); DFBResult (*WarpCursor) ( IDirectFBDisplayLayer *thiz, int x, int y ); DFBResult (*SetCursorAcceleration) ( IDirectFBDisplayLayer *thiz, int numerator, int denominator, int threshold ); DFBResult (*SetCursorShape) ( IDirectFBDisplayLayer *thiz, IDirectFBSurface *shape, int hot_x, int hot_y ); DFBResult (*SetCursorOpacity) ( IDirectFBDisplayLayer *thiz, u8 opacity ); DFBResult (*WaitForSync) ( IDirectFBDisplayLayer *thiz ); DFBResult (*SwitchContext) ( IDirectFBDisplayLayer *thiz, DFBBoolean exclusive ); DFBResult (*SetRotation) ( IDirectFBDisplayLayer *thiz, int rotation ); };
# 3036 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DSFLIP_NONE = 0x00000000,

     DSFLIP_WAIT = 0x00000001,

     DSFLIP_BLIT = 0x00000002,



     DSFLIP_ONSYNC = 0x00000004,



     DSFLIP_PIPELINE = 0x00000008,

     DSFLIP_WAITFORSYNC = DSFLIP_WAIT | DSFLIP_ONSYNC
} DFBSurfaceFlipFlags;




typedef enum {
     DSTF_LEFT = 0x00000000,
     DSTF_CENTER = 0x00000001,
     DSTF_RIGHT = 0x00000002,

     DSTF_TOP = 0x00000004,

     DSTF_BOTTOM = 0x00000008,


     DSTF_TOPLEFT = DSTF_TOP | DSTF_LEFT,
     DSTF_TOPCENTER = DSTF_TOP | DSTF_CENTER,
     DSTF_TOPRIGHT = DSTF_TOP | DSTF_RIGHT,

     DSTF_BOTTOMLEFT = DSTF_BOTTOM | DSTF_LEFT,
     DSTF_BOTTOMCENTER = DSTF_BOTTOM | DSTF_CENTER,
     DSTF_BOTTOMRIGHT = DSTF_BOTTOM | DSTF_RIGHT
} DFBSurfaceTextFlags;





typedef enum {
     DSLF_READ = 0x00000001,

     DSLF_WRITE = 0x00000002
} DFBSurfaceLockFlags;




typedef enum {



     DSPD_NONE = 0,
     DSPD_CLEAR = 1,
     DSPD_SRC = 2,
     DSPD_SRC_OVER = 3,
     DSPD_DST_OVER = 4,
     DSPD_SRC_IN = 5,
     DSPD_DST_IN = 6,
     DSPD_SRC_OUT = 7,
     DSPD_DST_OUT = 8,
     DSPD_SRC_ATOP = 9,
     DSPD_DST_ATOP = 10,
     DSPD_ADD = 11,
     DSPD_XOR = 12,
} DFBSurfacePorterDuffRule;




typedef enum {
     DSBF_UNKNOWN = 0,
     DSBF_ZERO = 1,
     DSBF_ONE = 2,
     DSBF_SRCCOLOR = 3,
     DSBF_INVSRCCOLOR = 4,
     DSBF_SRCALPHA = 5,
     DSBF_INVSRCALPHA = 6,
     DSBF_DESTALPHA = 7,
     DSBF_INVDESTALPHA = 8,
     DSBF_DESTCOLOR = 9,
     DSBF_INVDESTCOLOR = 10,
     DSBF_SRCALPHASAT = 11
} DFBSurfaceBlendFunction;




typedef struct {
     float x;
     float y;
     float z;
     float w;

     float s;
     float t;
} DFBVertex;




typedef enum {
     DTTF_LIST,
     DTTF_STRIP,
     DTTF_FAN
} DFBTriangleFormation;
# 3155 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBSurface { void *priv; int magic; DFBResult (*AddRef)( IDirectFBSurface *thiz ); DFBResult (*Release)( IDirectFBSurface *thiz ); DFBResult (*GetCapabilities) ( IDirectFBSurface *thiz, DFBSurfaceCapabilities *ret_caps ); DFBResult (*GetPosition) ( IDirectFBSurface *thiz, int *ret_x, int *ret_y ); DFBResult (*GetSize) ( IDirectFBSurface *thiz, int *ret_width, int *ret_height ); DFBResult (*GetVisibleRectangle) ( IDirectFBSurface *thiz, DFBRectangle *ret_rect ); DFBResult (*GetPixelFormat) ( IDirectFBSurface *thiz, DFBSurfacePixelFormat *ret_format ); DFBResult (*GetAccelerationMask) ( IDirectFBSurface *thiz, IDirectFBSurface *source, DFBAccelerationMask *ret_mask ); DFBResult (*GetPalette) ( IDirectFBSurface *thiz, IDirectFBPalette **ret_interface ); DFBResult (*SetPalette) ( IDirectFBSurface *thiz, IDirectFBPalette *palette ); DFBResult (*SetAlphaRamp) ( IDirectFBSurface *thiz, u8 a0, u8 a1, u8 a2, u8 a3 ); DFBResult (*Lock) ( IDirectFBSurface *thiz, DFBSurfaceLockFlags flags, void **ret_ptr, int *ret_pitch ); DFBResult (*GetFramebufferOffset) ( IDirectFBSurface *thiz, int *offset ); DFBResult (*Unlock) ( IDirectFBSurface *thiz ); DFBResult (*Flip) ( IDirectFBSurface *thiz, const DFBRegion *region, DFBSurfaceFlipFlags flags ); DFBResult (*SetField) ( IDirectFBSurface *thiz, int field ); DFBResult (*Clear) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*SetClip) ( IDirectFBSurface *thiz, const DFBRegion *clip ); DFBResult (*GetClip) ( IDirectFBSurface *thiz, DFBRegion *ret_clip ); DFBResult (*SetColor) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b, u8 a ); DFBResult (*SetColorIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetSrcBlendFunction) ( IDirectFBSurface *thiz, DFBSurfaceBlendFunction function ); DFBResult (*SetDstBlendFunction) ( IDirectFBSurface *thiz, DFBSurfaceBlendFunction function ); DFBResult (*SetPorterDuff) ( IDirectFBSurface *thiz, DFBSurfacePorterDuffRule rule ); DFBResult (*SetSrcColorKey) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetSrcColorKeyIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetDstColorKey) ( IDirectFBSurface *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetDstColorKeyIndex) ( IDirectFBSurface *thiz, unsigned int index ); DFBResult (*SetBlittingFlags) ( IDirectFBSurface *thiz, DFBSurfaceBlittingFlags flags ); DFBResult (*Blit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, int x, int y ); DFBResult (*TileBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, int x, int y ); DFBResult (*BatchBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rects, const DFBPoint *dest_points, int num ); DFBResult (*StretchBlit) ( IDirectFBSurface *thiz, IDirectFBSurface *source, const DFBRectangle *source_rect, const DFBRectangle *destination_rect ); DFBResult (*TextureTriangles) ( IDirectFBSurface *thiz, IDirectFBSurface *texture, const DFBVertex *vertices, const int *indices, int num, DFBTriangleFormation formation ); DFBResult (*SetDrawingFlags) ( IDirectFBSurface *thiz, DFBSurfaceDrawingFlags flags ); DFBResult (*FillRectangle) ( IDirectFBSurface *thiz, int x, int y, int w, int h ); DFBResult (*DrawRectangle) ( IDirectFBSurface *thiz, int x, int y, int w, int h ); DFBResult (*DrawLine) ( IDirectFBSurface *thiz, int x1, int y1, int x2, int y2 ); DFBResult (*DrawLines) ( IDirectFBSurface *thiz, const DFBRegion *lines, unsigned int num_lines ); DFBResult (*FillTriangle) ( IDirectFBSurface *thiz, int x1, int y1, int x2, int y2, int x3, int y3 ); DFBResult (*FillRectangles) ( IDirectFBSurface *thiz, const DFBRectangle *rects, unsigned int num ); DFBResult (*FillSpans) ( IDirectFBSurface *thiz, int y, const DFBSpan *spans, unsigned int num ); DFBResult (*SetFont) ( IDirectFBSurface *thiz, IDirectFBFont *font ); DFBResult (*GetFont) ( IDirectFBSurface *thiz, IDirectFBFont **ret_font ); DFBResult (*DrawString) ( IDirectFBSurface *thiz, const char *text, int bytes, int x, int y, DFBSurfaceTextFlags flags ); DFBResult (*DrawGlyph) ( IDirectFBSurface *thiz, unsigned int character, int x, int y, DFBSurfaceTextFlags flags ); DFBResult (*SetEncoding) ( IDirectFBSurface *thiz, DFBTextEncodingID encoding ); DFBResult (*GetSubSurface) ( IDirectFBSurface *thiz, const DFBRectangle *rect, IDirectFBSurface **ret_interface ); DFBResult (*GetGL) ( IDirectFBSurface *thiz, IDirectFBGL **ret_interface ); DFBResult (*Dump) ( IDirectFBSurface *thiz, const char *directory, const char *prefix ); DFBResult (*DisableAcceleration) ( IDirectFBSurface *thiz, DFBAccelerationMask mask ); DFBResult (*ReleaseSource) ( IDirectFBSurface *thiz ); DFBResult (*SetIndexTranslation) ( IDirectFBSurface *thiz, const int *indices, int num_indices ); DFBResult (*SetRenderOptions) ( IDirectFBSurface *thiz, DFBSurfaceRenderOptions options ); DFBResult (*SetMatrix) ( IDirectFBSurface *thiz, const s32 *matrix ); };
# 3869 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBPalette { void *priv; int magic; DFBResult (*AddRef)( IDirectFBPalette *thiz ); DFBResult (*Release)( IDirectFBPalette *thiz ); DFBResult (*GetCapabilities) ( IDirectFBPalette *thiz, DFBPaletteCapabilities *ret_caps ); DFBResult (*GetSize) ( IDirectFBPalette *thiz, unsigned int *ret_size ); DFBResult (*SetEntries) ( IDirectFBPalette *thiz, const DFBColor *entries, unsigned int num_entries, unsigned int offset ); DFBResult (*GetEntries) ( IDirectFBPalette *thiz, DFBColor *ret_entries, unsigned int num_entries, unsigned int offset ); DFBResult (*FindBestMatch) ( IDirectFBPalette *thiz, u8 r, u8 g, u8 b, u8 a, unsigned int *ret_index ); DFBResult (*CreateCopy) ( IDirectFBPalette *thiz, IDirectFBPalette **ret_interface ); DFBResult (*SetEntriesYUV) ( IDirectFBPalette *thiz, const DFBColorYUV *entries, unsigned int num_entries, unsigned int offset ); DFBResult (*GetEntriesYUV) ( IDirectFBPalette *thiz, DFBColorYUV *ret_entries, unsigned int num_entries, unsigned int offset ); DFBResult (*FindBestMatchYUV) ( IDirectFBPalette *thiz, u8 y, u8 u, u8 v, u8 a, unsigned int *ret_index ); };
# 3991 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DIKS_UP = 0x00000000,
     DIKS_DOWN = 0x00000001
} DFBInputDeviceKeyState;




typedef enum {
     DIBS_UP = 0x00000000,
     DIBS_DOWN = 0x00000001
} DFBInputDeviceButtonState;




typedef enum {
     DIBM_LEFT = 0x00000001,
     DIBM_RIGHT = 0x00000002,
     DIBM_MIDDLE = 0x00000004
} DFBInputDeviceButtonMask;




typedef enum {
     DIMM_SHIFT = (1 << DIMKI_SHIFT),
     DIMM_CONTROL = (1 << DIMKI_CONTROL),
     DIMM_ALT = (1 << DIMKI_ALT),
     DIMM_ALTGR = (1 << DIMKI_ALTGR),
     DIMM_META = (1 << DIMKI_META),
     DIMM_SUPER = (1 << DIMKI_SUPER),
     DIMM_HYPER = (1 << DIMKI_HYPER)
} DFBInputDeviceModifierMask;
# 4034 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBInputDevice { void *priv; int magic; DFBResult (*AddRef)( IDirectFBInputDevice *thiz ); DFBResult (*Release)( IDirectFBInputDevice *thiz ); DFBResult (*GetID) ( IDirectFBInputDevice *thiz, DFBInputDeviceID *ret_device_id ); DFBResult (*GetDescription) ( IDirectFBInputDevice *thiz, DFBInputDeviceDescription *ret_desc ); DFBResult (*GetKeymapEntry) ( IDirectFBInputDevice *thiz, int keycode, DFBInputDeviceKeymapEntry *ret_entry ); DFBResult (*CreateEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*DetachEventBuffer) ( IDirectFBInputDevice *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*GetKeyState) ( IDirectFBInputDevice *thiz, DFBInputDeviceKeyIdentifier key_id, DFBInputDeviceKeyState *ret_state ); DFBResult (*GetModifiers) ( IDirectFBInputDevice *thiz, DFBInputDeviceModifierMask *ret_modifiers ); DFBResult (*GetLockState) ( IDirectFBInputDevice *thiz, DFBInputDeviceLockState *ret_locks ); DFBResult (*GetButtons) ( IDirectFBInputDevice *thiz, DFBInputDeviceButtonMask *ret_buttons ); DFBResult (*GetButtonState) ( IDirectFBInputDevice *thiz, DFBInputDeviceButtonIdentifier button, DFBInputDeviceButtonState *ret_state ); DFBResult (*GetAxis) ( IDirectFBInputDevice *thiz, DFBInputDeviceAxisIdentifier axis, int *ret_pos ); DFBResult (*GetXY) ( IDirectFBInputDevice *thiz, int *ret_x, int *ret_y ); };
# 4171 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DFEC_NONE = 0x00,
     DFEC_INPUT = 0x01,
     DFEC_WINDOW = 0x02,
     DFEC_USER = 0x03,
     DFEC_UNIVERSAL = 0x04,
     DFEC_VIDEOPROVIDER = 0x05
} DFBEventClass;




typedef enum {
     DIET_UNKNOWN = 0,
     DIET_KEYPRESS,
     DIET_KEYRELEASE,
     DIET_BUTTONPRESS,
     DIET_BUTTONRELEASE,
     DIET_AXISMOTION
} DFBInputEventType;




typedef enum {
     DIEF_NONE = 0x000,
     DIEF_TIMESTAMP = 0x001,
     DIEF_AXISABS = 0x002,
     DIEF_AXISREL = 0x004,

     DIEF_KEYCODE = 0x008,

     DIEF_KEYID = 0x010,

     DIEF_KEYSYMBOL = 0x020,

     DIEF_MODIFIERS = 0x040,

     DIEF_LOCKS = 0x080,

     DIEF_BUTTONS = 0x100,

     DIEF_GLOBAL = 0x200,





     DIEF_REPEAT = 0x400,
     DIEF_FOLLOW = 0x800
} DFBInputEventFlags;




typedef struct {
     DFBEventClass clazz;

     DFBInputEventType type;
     DFBInputDeviceID device_id;
     DFBInputEventFlags flags;



     struct timeval timestamp;


     int key_code;



     DFBInputDeviceKeyIdentifier key_id;

     DFBInputDeviceKeySymbol key_symbol;



     DFBInputDeviceModifierMask modifiers;

     DFBInputDeviceLockState locks;



     DFBInputDeviceButtonIdentifier button;

     DFBInputDeviceButtonMask buttons;



     DFBInputDeviceAxisIdentifier axis;


     int axisabs;

     int axisrel;

} DFBInputEvent;




typedef enum {
     DWET_NONE = 0x00000000,

     DWET_POSITION = 0x00000001,


     DWET_SIZE = 0x00000002,


     DWET_CLOSE = 0x00000004,

     DWET_DESTROYED = 0x00000008,


     DWET_GOTFOCUS = 0x00000010,
     DWET_LOSTFOCUS = 0x00000020,

     DWET_KEYDOWN = 0x00000100,

     DWET_KEYUP = 0x00000200,


     DWET_BUTTONDOWN = 0x00010000,

     DWET_BUTTONUP = 0x00020000,

     DWET_MOTION = 0x00040000,

     DWET_ENTER = 0x00080000,

     DWET_LEAVE = 0x00100000,

     DWET_WHEEL = 0x00200000,


     DWET_POSITION_SIZE = DWET_POSITION | DWET_SIZE,



     DWET_ALL = 0x003F033F
} DFBWindowEventType;




typedef enum {
     DWEF_NONE = 0x00000000,

     DWEF_RETURNED = 0x00000001,

     DWEF_ALL = 0x00000001
} DFBWindowEventFlags;




typedef enum {
     DVPET_NONE = 0x00000000,
     DVPET_STARTED = 0x00000001,
     DVPET_STOPPED = 0x00000002,
     DVPET_SPEEDCHANGE = 0x00000004,
     DVPET_STREAMCHANGE = 0x00000008,
     DVPET_FATALERROR = 0x00000010,
     DVPET_FINISHED = 0x00000020,
     DVPET_SURFACECHANGE = 0x00000040,
     DVPET_ALL = 0x0000007F
} DFBVideoProviderEventType;




typedef struct {
     DFBEventClass clazz;

     DFBWindowEventType type;
     DFBWindowEventFlags flags;

     DFBWindowID window_id;



     int x;


     int y;





     int cx;
     int cy;


     int step;


     int w;
     int h;


     int key_code;



     DFBInputDeviceKeyIdentifier key_id;

     DFBInputDeviceKeySymbol key_symbol;


     DFBInputDeviceModifierMask modifiers;
     DFBInputDeviceLockState locks;


     DFBInputDeviceButtonIdentifier button;


     DFBInputDeviceButtonMask buttons;


     struct timeval timestamp;
} DFBWindowEvent;




typedef struct {
     DFBEventClass clazz;

     DFBVideoProviderEventType type;
} DFBVideoProviderEvent;




typedef struct {
     DFBEventClass clazz;

     unsigned int type;
     void *data;
} DFBUserEvent;




typedef struct {
     DFBEventClass clazz;
     unsigned int size;




} DFBUniversalEvent;




typedef union {
     DFBEventClass clazz;
     DFBInputEvent input;
     DFBWindowEvent window;
     DFBUserEvent user;
     DFBUniversalEvent universal;
     DFBVideoProviderEvent videoprovider;
} DFBEvent;






typedef struct {
     unsigned int num_events;

     unsigned int DFEC_INPUT;
     unsigned int DFEC_WINDOW;
     unsigned int DFEC_USER;
     unsigned int DFEC_UNIVERSAL;
     unsigned int DFEC_VIDEOPROVIDER;

     unsigned int DIET_KEYPRESS;
     unsigned int DIET_KEYRELEASE;
     unsigned int DIET_BUTTONPRESS;
     unsigned int DIET_BUTTONRELEASE;
     unsigned int DIET_AXISMOTION;

     unsigned int DWET_POSITION;
     unsigned int DWET_SIZE;
     unsigned int DWET_CLOSE;
     unsigned int DWET_DESTROYED;
     unsigned int DWET_GOTFOCUS;
     unsigned int DWET_LOSTFOCUS;
     unsigned int DWET_KEYDOWN;
     unsigned int DWET_KEYUP;
     unsigned int DWET_BUTTONDOWN;
     unsigned int DWET_BUTTONUP;
     unsigned int DWET_MOTION;
     unsigned int DWET_ENTER;
     unsigned int DWET_LEAVE;
     unsigned int DWET_WHEEL;
     unsigned int DWET_POSITION_SIZE;

     unsigned int DVPET_STARTED;
     unsigned int DVPET_STOPPED;
     unsigned int DVPET_SPEEDCHANGE;
     unsigned int DVPET_STREAMCHANGE;
     unsigned int DVPET_FATALERROR;
     unsigned int DVPET_FINISHED;
     unsigned int DVPET_SURFACECHANGE;
} DFBEventBufferStats;
# 4491 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBEventBuffer { void *priv; int magic; DFBResult (*AddRef)( IDirectFBEventBuffer *thiz ); DFBResult (*Release)( IDirectFBEventBuffer *thiz ); DFBResult (*Reset) ( IDirectFBEventBuffer *thiz ); DFBResult (*WaitForEvent) ( IDirectFBEventBuffer *thiz ); DFBResult (*WaitForEventWithTimeout) ( IDirectFBEventBuffer *thiz, unsigned int seconds, unsigned int milli_seconds ); DFBResult (*GetEvent) ( IDirectFBEventBuffer *thiz, DFBEvent *ret_event ); DFBResult (*PeekEvent) ( IDirectFBEventBuffer *thiz, DFBEvent *ret_event ); DFBResult (*HasEvent) ( IDirectFBEventBuffer *thiz ); DFBResult (*PostEvent) ( IDirectFBEventBuffer *thiz, const DFBEvent *event ); DFBResult (*WakeUp) ( IDirectFBEventBuffer *thiz ); DFBResult (*CreateFileDescriptor) ( IDirectFBEventBuffer *thiz, int *ret_fd ); DFBResult (*EnableStatistics) ( IDirectFBEventBuffer *thiz, DFBBoolean enable ); DFBResult (*GetStatistics) ( IDirectFBEventBuffer *thiz, DFBEventBufferStats *ret_stats ); };
# 4622 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DWKS_ALL = 0x00000000,
     DWKS_NONE = 0x00000001,
     DWKS_LIST = 0x00000002
} DFBWindowKeySelection;

typedef enum {
     DWGM_DEFAULT = 0x00000000,
     DWGM_FOLLOW = 0x00000001,
     DWGM_RECTANGLE = 0x00000002,
     DWGM_LOCATION = 0x00000003
} DFBWindowGeometryMode;

typedef struct {
     DFBWindowGeometryMode mode;

     DFBRectangle rectangle;
     DFBLocation location;
} DFBWindowGeometry;
# 4649 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBWindow { void *priv; int magic; DFBResult (*AddRef)( IDirectFBWindow *thiz ); DFBResult (*Release)( IDirectFBWindow *thiz ); DFBResult (*GetID) ( IDirectFBWindow *thiz, DFBWindowID *ret_window_id ); DFBResult (*GetPosition) ( IDirectFBWindow *thiz, int *ret_x, int *ret_y ); DFBResult (*GetSize) ( IDirectFBWindow *thiz, int *ret_width, int *ret_height ); DFBResult (*CreateEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*DetachEventBuffer) ( IDirectFBWindow *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*EnableEvents) ( IDirectFBWindow *thiz, DFBWindowEventType mask ); DFBResult (*DisableEvents) ( IDirectFBWindow *thiz, DFBWindowEventType mask ); DFBResult (*GetSurface) ( IDirectFBWindow *thiz, IDirectFBSurface **ret_surface ); DFBResult (*SetProperty) ( IDirectFBWindow *thiz, const char *key, void *value, void **ret_old_value ); DFBResult (*GetProperty) ( IDirectFBWindow *thiz, const char *key, void **ret_value ); DFBResult (*RemoveProperty) ( IDirectFBWindow *thiz, const char *key, void **ret_value ); DFBResult (*SetOptions) ( IDirectFBWindow *thiz, DFBWindowOptions options ); DFBResult (*GetOptions) ( IDirectFBWindow *thiz, DFBWindowOptions *ret_options ); DFBResult (*SetColorKey) ( IDirectFBWindow *thiz, u8 r, u8 g, u8 b ); DFBResult (*SetColorKeyIndex) ( IDirectFBWindow *thiz, unsigned int index ); DFBResult (*SetOpacity) ( IDirectFBWindow *thiz, u8 opacity ); DFBResult (*SetOpaqueRegion) ( IDirectFBWindow *thiz, int x1, int y1, int x2, int y2 ); DFBResult (*GetOpacity) ( IDirectFBWindow *thiz, u8 *ret_opacity ); DFBResult (*SetCursorShape) ( IDirectFBWindow *thiz, IDirectFBSurface *shape, int hot_x, int hot_y ); DFBResult (*RequestFocus) ( IDirectFBWindow *thiz ); DFBResult (*GrabKeyboard) ( IDirectFBWindow *thiz ); DFBResult (*UngrabKeyboard) ( IDirectFBWindow *thiz ); DFBResult (*GrabPointer) ( IDirectFBWindow *thiz ); DFBResult (*UngrabPointer) ( IDirectFBWindow *thiz ); DFBResult (*GrabKey) ( IDirectFBWindow *thiz, DFBInputDeviceKeySymbol symbol, DFBInputDeviceModifierMask modifiers ); DFBResult (*UngrabKey) ( IDirectFBWindow *thiz, DFBInputDeviceKeySymbol symbol, DFBInputDeviceModifierMask modifiers ); DFBResult (*Move) ( IDirectFBWindow *thiz, int dx, int dy ); DFBResult (*MoveTo) ( IDirectFBWindow *thiz, int x, int y ); DFBResult (*Resize) ( IDirectFBWindow *thiz, int width, int height ); DFBResult (*SetStackingClass) ( IDirectFBWindow *thiz, DFBWindowStackingClass stacking_class ); DFBResult (*Raise) ( IDirectFBWindow *thiz ); DFBResult (*Lower) ( IDirectFBWindow *thiz ); DFBResult (*RaiseToTop) ( IDirectFBWindow *thiz ); DFBResult (*LowerToBottom) ( IDirectFBWindow *thiz ); DFBResult (*PutAtop) ( IDirectFBWindow *thiz, IDirectFBWindow *lower ); DFBResult (*PutBelow) ( IDirectFBWindow *thiz, IDirectFBWindow *upper ); DFBResult (*Close) ( IDirectFBWindow *thiz ); DFBResult (*Destroy) ( IDirectFBWindow *thiz ); DFBResult (*SetBounds) ( IDirectFBWindow *thiz, int x, int y, int width, int height ); DFBResult (*ResizeSurface) ( IDirectFBWindow *thiz, int width, int height ); DFBResult (*Bind) ( IDirectFBWindow *thiz, IDirectFBWindow *window, int x, int y ); DFBResult (*Unbind) ( IDirectFBWindow *thiz, IDirectFBWindow *window ); DFBResult (*SetKeySelection) ( IDirectFBWindow *thiz, DFBWindowKeySelection selection, const DFBInputDeviceKeySymbol *keys, unsigned int num_keys ); DFBResult (*GrabUnselectedKeys) ( IDirectFBWindow *thiz ); DFBResult (*UngrabUnselectedKeys) ( IDirectFBWindow *thiz ); DFBResult (*SetSrcGeometry) ( IDirectFBWindow *thiz, const DFBWindowGeometry *geometry ); DFBResult (*SetDstGeometry) ( IDirectFBWindow *thiz, const DFBWindowGeometry *geometry ); };
# 5169 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef DFBEnumerationResult (*DFBTextEncodingCallback) (
     DFBTextEncodingID encoding_id,
     const char *name,
     void *context
);
# 5182 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBFont { void *priv; int magic; DFBResult (*AddRef)( IDirectFBFont *thiz ); DFBResult (*Release)( IDirectFBFont *thiz ); DFBResult (*GetAscender) ( IDirectFBFont *thiz, int *ret_ascender ); DFBResult (*GetDescender) ( IDirectFBFont *thiz, int *ret_descender ); DFBResult (*GetHeight) ( IDirectFBFont *thiz, int *ret_height ); DFBResult (*GetMaxAdvance) ( IDirectFBFont *thiz, int *ret_maxadvance ); DFBResult (*GetKerning) ( IDirectFBFont *thiz, unsigned int prev, unsigned int current, int *ret_kern_x, int *ret_kern_y ); DFBResult (*GetStringWidth) ( IDirectFBFont *thiz, const char *text, int bytes, int *ret_width ); DFBResult (*GetStringExtents) ( IDirectFBFont *thiz, const char *text, int bytes, DFBRectangle *ret_logical_rect, DFBRectangle *ret_ink_rect ); DFBResult (*GetGlyphExtents) ( IDirectFBFont *thiz, unsigned int character, DFBRectangle *ret_rect, int *ret_advance ); DFBResult (*GetStringBreak) ( IDirectFBFont *thiz, const char *text, int bytes, int max_width, int *ret_width, int *ret_str_length, const char **ret_next_line ); DFBResult (*SetEncoding) ( IDirectFBFont *thiz, DFBTextEncodingID encoding ); DFBResult (*EnumEncodings) ( IDirectFBFont *thiz, DFBTextEncodingCallback callback, void *context ); DFBResult (*FindEncoding) ( IDirectFBFont *thiz, const char *name, DFBTextEncodingID *ret_encoding ); };
# 5379 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DICAPS_NONE = 0x00000000,
     DICAPS_ALPHACHANNEL = 0x00000001,

     DICAPS_COLORKEY = 0x00000002


} DFBImageCapabilities;





typedef struct {
     DFBImageCapabilities caps;

     u8 colorkey_r;
     u8 colorkey_g;
     u8 colorkey_b;
} DFBImageDescription;


typedef enum {
        DIRCR_OK,
        DIRCR_ABORT
} DIRenderCallbackResult;




typedef DIRenderCallbackResult (*DIRenderCallback)(DFBRectangle *rect, void *ctx);
# 5418 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBImageProvider { void *priv; int magic; DFBResult (*AddRef)( IDirectFBImageProvider *thiz ); DFBResult (*Release)( IDirectFBImageProvider *thiz ); DFBResult (*GetSurfaceDescription) ( IDirectFBImageProvider *thiz, DFBSurfaceDescription *ret_dsc ); DFBResult (*GetImageDescription) ( IDirectFBImageProvider *thiz, DFBImageDescription *ret_dsc ); DFBResult (*RenderTo) ( IDirectFBImageProvider *thiz, IDirectFBSurface *destination, const DFBRectangle *destination_rect ); DFBResult (*SetRenderCallback) ( IDirectFBImageProvider *thiz, DIRenderCallback callback, void *callback_data ); };
# 5484 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef enum {
     DVSCAPS_NONE = 0x00000000,
     DVSCAPS_VIDEO = 0x00000001,
     DVSCAPS_AUDIO = 0x00000002

} DFBStreamCapabilities;
# 5501 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
typedef struct {
     DFBStreamCapabilities caps;

     struct {
          char encoding[30];

          double framerate;
          double aspect;
          int bitrate;
       int afd;
       int width;
       int height;
     }
      video;

     struct {
          char encoding[30];

          int samplerate;
          int channels;
          int bitrate;
     }
      audio;

     char title[255];
     char author[255];
     char album[255];
     short year;
     char genre[32];
     char comment[255];
} DFBStreamDescription;




typedef enum {
     DSF_ES = 0x00000000,
     DSF_PES = 0x00000001,
} DFBStreamFormat;




typedef struct {
     struct {
          char encoding[30];

          DFBStreamFormat format;

     } video;

     struct {
          char encoding[30];

          DFBStreamFormat format;
     } audio;
} DFBStreamAttributes;




typedef void (*DVFrameCallback)(void *ctx);
# 5572 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBVideoProvider { void *priv; int magic; DFBResult (*AddRef)( IDirectFBVideoProvider *thiz ); DFBResult (*Release)( IDirectFBVideoProvider *thiz ); DFBResult (*GetCapabilities) ( IDirectFBVideoProvider *thiz, DFBVideoProviderCapabilities *ret_caps ); DFBResult (*GetSurfaceDescription) ( IDirectFBVideoProvider *thiz, DFBSurfaceDescription *ret_dsc ); DFBResult (*GetStreamDescription) ( IDirectFBVideoProvider *thiz, DFBStreamDescription *ret_dsc ); DFBResult (*PlayTo) ( IDirectFBVideoProvider *thiz, IDirectFBSurface *destination, const DFBRectangle *destination_rect, DVFrameCallback callback, void *ctx ); DFBResult (*Stop) ( IDirectFBVideoProvider *thiz ); DFBResult (*GetStatus) ( IDirectFBVideoProvider *thiz, DFBVideoProviderStatus *ret_status ); DFBResult (*SeekTo) ( IDirectFBVideoProvider *thiz, double seconds ); DFBResult (*GetPos) ( IDirectFBVideoProvider *thiz, double *ret_seconds ); DFBResult (*GetLength) ( IDirectFBVideoProvider *thiz, double *ret_seconds ); DFBResult (*GetColorAdjustment) ( IDirectFBVideoProvider *thiz, DFBColorAdjustment *ret_adj ); DFBResult (*SetColorAdjustment) ( IDirectFBVideoProvider *thiz, const DFBColorAdjustment *adj ); DFBResult (*SendEvent) ( IDirectFBVideoProvider *thiz, const DFBEvent *event ); DFBResult (*SetPlaybackFlags) ( IDirectFBVideoProvider *thiz, DFBVideoProviderPlaybackFlags flags ); DFBResult (*SetSpeed) ( IDirectFBVideoProvider *thiz, double multiplier ); DFBResult (*GetSpeed) ( IDirectFBVideoProvider *thiz, double *ret_multiplier ); DFBResult (*SetVolume) ( IDirectFBVideoProvider *thiz, float level ); DFBResult (*GetVolume) ( IDirectFBVideoProvider *thiz, float *ret_level ); DFBResult (*SetStreamAttributes) ( IDirectFBVideoProvider *thiz, DFBStreamDescription attr ); DFBResult (*SetAudioOutputs) ( IDirectFBVideoProvider *thiz, DFBVideoProviderAudioUnits *audioUnits ); DFBResult (*GetAudioOutputs) ( IDirectFBVideoProvider *thiz, DFBVideoProviderAudioUnits *audioUnits ); DFBResult (*CreateEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer **ret_buffer ); DFBResult (*AttachEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer *buffer ); DFBResult (*EnableEvents) ( IDirectFBVideoProvider *thiz, DFBVideoProviderEventType mask ); DFBResult (*DisableEvents) ( IDirectFBVideoProvider *thiz, DFBVideoProviderEventType mask ); DFBResult (*DetachEventBuffer) ( IDirectFBVideoProvider *thiz, IDirectFBEventBuffer *buffer ); };
# 5835 "/home/lucas/software/pr11/stb225_tarballs/DirectFB-1.1.1_nxp/include/directfb.h"
struct _IDirectFBDataBuffer { void *priv; int magic; DFBResult (*AddRef)( IDirectFBDataBuffer *thiz ); DFBResult (*Release)( IDirectFBDataBuffer *thiz ); DFBResult (*Flush) ( IDirectFBDataBuffer *thiz ); DFBResult (*Finish) ( IDirectFBDataBuffer *thiz ); DFBResult (*SeekTo) ( IDirectFBDataBuffer *thiz, unsigned int offset ); DFBResult (*GetPosition) ( IDirectFBDataBuffer *thiz, unsigned int *ret_offset ); DFBResult (*GetLength) ( IDirectFBDataBuffer *thiz, unsigned int *ret_length ); DFBResult (*WaitForData) ( IDirectFBDataBuffer *thiz, unsigned int length ); DFBResult (*WaitForDataWithTimeout) ( IDirectFBDataBuffer *thiz, unsigned int length, unsigned int seconds, unsigned int milli_seconds ); DFBResult (*GetData) ( IDirectFBDataBuffer *thiz, unsigned int length, void *ret_data, unsigned int *ret_read ); DFBResult (*PeekData) ( IDirectFBDataBuffer *thiz, unsigned int length, int offset, void *ret_data, unsigned int *ret_read ); DFBResult (*HasData) ( IDirectFBDataBuffer *thiz ); DFBResult (*PutData) ( IDirectFBDataBuffer *thiz, const void *data, unsigned int length ); DFBResult (*CreateImageProvider) ( IDirectFBDataBuffer *thiz, IDirectFBImageProvider **interface ); DFBResult (*CreateVideoProvider) ( IDirectFBDataBuffer *thiz, IDirectFBVideoProvider **interface ); };
# 75 "exStbKey.c" 2
# 118 "exStbKey.c"
typedef struct rcLookup_tag
{
    int32_t DFBCode;
    const char *name;
} rcLookUp;
# 211 "exStbKey.c"
static const char *noName = "No Name";

static const rcLookUp keySymbolCodes[] = {
     { DIKS_0, "Digit 0" },
     { DIKS_1, "Digit 1" },
     { DIKS_2, "Digit 2" },
     { DIKS_3, "Digit 3" },
     { DIKS_4, "Digit 4" },
     { DIKS_5, "Digit 5" },
     { DIKS_6, "Digit 6" },
     { DIKS_7, "Digit 7" },
     { DIKS_8, "Digit 8" },
     { DIKS_9, "Digit 9" },
     { DIKS_DIGITS, "Digit Select" },
     { DIKS_OPTION, "Option" },
     { DIKS_POWER, "Power" },
     { DIKS_MUTE, "Mute" },
     { ((DIKT_CUSTOM) | ((0))), "Personal" },
     { ((DIKT_CUSTOM) | ((1))), "Info" },
     { DIKS_VOLUME_UP, "Volume Up" },
     { DIKS_VOLUME_DOWN, "Volume Down" },
     { DIKS_RECORD, "Record" },
     { DIKS_CHANNEL_UP, "Channel Up" },
     { DIKS_CHANNEL_DOWN, "Channel Down" },
     { DIKS_PREVIOUS, "Previous" },
     { DIKS_LANGUAGE, "Language" },
     { ((DIKT_CUSTOM) | ((2))), "Audio" },
     { ((DIKT_CUSTOM) | ((3))), "Hold" },
     { ((DIKT_CUSTOM) | ((4))), "Clock" },
     { ((DIKT_CUSTOM) | ((5))), "Expand" },
     { ((DIKT_CUSTOM) | ((6))), "Reveal" },
     { ((DIKT_CUSTOM) | ((7))), "Cancel" },
     { ((DIKT_CUSTOM) | ((8))), "Mix" },
     { DIKS_PLAYPAUSE, "Play/Pause" },
     { DIKS_PLAY, "Play" },
     { DIKS_FORWARD, "Forward" },
     { DIKS_FASTFORWARD, "Fast Forward" },
     { DIKS_PAUSE, "Pause" },
     { DIKS_STOP, "Stop" },
     { DIKS_REWIND, "Rewind" },
     { DIKS_AUX, "Aux Input" },
     { DIKS_TEXT, "Text" },
     { DIKS_TV, "TV" },
     { DIKS_CURSOR_UP, "Cursor Up" },
     { DIKS_CURSOR_DOWN, "Cursor Down" },
     { DIKS_MENU, "Menu" },
     { DIKS_CURSOR_LEFT, "Cursor Left" },
     { DIKS_CURSOR_RIGHT, "Cursor Right" },
     { ((DIKT_CUSTOM) | ((9))), "PIP" },
     { ((DIKT_CUSTOM) | ((10))), "PIP Move" },
     { ((DIKT_CUSTOM) | ((11))), "Picture Improve" },
     { DIKS_CHANNEL_UP, "Channel Up" },
     { DIKS_CHANNEL_DOWN, "Channel Down" },
     { DIKS_RED, "Red" },
     { DIKS_GREEN, "Green" },
     { DIKS_YELLOW, "Yellow" },
     { DIKS_BLUE, "Blue" },
     { ((DIKT_CUSTOM) | ((12))), "White" },
     { DIKS_SUBTITLE, "Subtitle" },
     { ((DIKT_CUSTOM) | ((13))), "Store" },
     { ((DIKT_CUSTOM) | ((14))), "Widescreen" },
     { ((DIKT_CUSTOM) | ((15))), "PAP" },
     { ((DIKT_CUSTOM) | ((16))), "Smart Sound" },
     { ((DIKT_CUSTOM) | ((17))),"Active Control" },
     { ((DIKT_CUSTOM) | ((18))), "Smart Picture" },
     { DIKS_EPG, "EPG" },
     { DIKS_RADIO, "Radio" },
     { DIKS_EXIT, "Exit" },
     { DIKS_OK, "OK" },
     { DIKS_CHANNEL, "All Channels" },
     { DIKS_ASTERISK, "Star" },
     { ((DIKT_CUSTOM) | ((21))), "Save" },
     { DIKS_HELP, "Help" },
     { DIKS_ENTER, "Enter" },
     { ((DIKT_CUSTOM) | ((22))), "Lock" },
     { DIKS_LAST, "Last" },
     { DIKS_NEXT, "Next" },
     { ((DIKT_CUSTOM) | ((23))), "Slow Forwards" },
     { ((DIKT_CUSTOM) | ((24))), "Slow Backwards" },
     { ((DIKT_CUSTOM) | ((25))), "Skip Forwards" },
     { ((DIKT_CUSTOM) | ((26))), "Skip Backwards" },
     { DIKS_EJECT, "Eject" },
     { ((DIKT_CUSTOM) | ((30))), "More" },
     { DIKS_FAVORITES, "Favourites" },
     { DIKS_NUMBER_SIGN, "Hash" },
     { DIKS_SCREEN, "Screen" },
     { DIKS_LIST, "List" },
     { DIKS_TIME, "Time" },
     { DIKS_INFO, "Info" },
     { DIKS_NUMBER_SIGN, "Number sign" },
     { DIKS_ARCHIVE, "Archive" },
     { DIKS_TITLE, "Title" },
     { DIKS_ZOOM, "Zoom" },
     { DIKS_BACK, "Back" },
     { DIKS_SETUP, "Setup" },
     { DIKS_DVD, "DVD" },
     { DIKS_PVR, "PVR" },
     { DIKS_AUDIO, "Audio" },
     { DIKS_INTERNET, "Internet" },
     { DIKS_PC, "PC" },
     { DIKS_ANGLE, "Angle" }
};

static const rcLookUp keyIdentifierCodes[] = {
     { DIKI_A, "a"},
     { DIKI_B, "b"},
     { DIKI_C, "c"},
     { DIKI_D, "d"},
     { DIKI_E, "e"},
     { DIKI_F, "f"},
     { DIKI_G, "g"},
     { DIKI_H, "h"},
     { DIKI_I, "i"},
     { DIKI_J, "j"},
     { DIKI_K, "k"},
     { DIKI_L, "l"},
     { DIKI_M, "m"},
     { DIKI_N, "n"},
     { DIKI_O, "o"},
     { DIKI_P, "p"},
     { DIKI_Q, "q"},
     { DIKI_R, "r"},
     { DIKI_S, "s"},
     { DIKI_T, "t"},
     { DIKI_U, "u"},
     { DIKI_V, "v"},
     { DIKI_W, "w"},
     { DIKI_X, "x"},
     { DIKI_Y, "y"},
     { DIKI_Z, "z"},
     { DIKI_0, "0"},
     { DIKI_1, "1"},
     { DIKI_2, "2"},
     { DIKI_3, "3"},
     { DIKI_4, "4"},
     { DIKI_5, "5"},
     { DIKI_6, "6"},
     { DIKI_7, "7"},
     { DIKI_8, "8"},
     { DIKI_9, "9"},
     { DIKI_F1, "F1"},
     { DIKI_F2, "F2"},
     { DIKI_F3, "F3"},
     { DIKI_F4, "F4"},
     { DIKI_F5, "F5"},
     { DIKI_F6, "F6"},
     { DIKI_F7, "F7"},
     { DIKI_F8, "F8"},
     { DIKI_F9, "F9"},
     { DIKI_F10, "F10"},
     { DIKI_SHIFT_L, "Shift Left"},
     { DIKI_SHIFT_R, "Shift Right"},
     { DIKI_CONTROL_L, "Control Left"},
     { DIKI_ALT_L, "Alt Left"},
     { DIKI_ALT_R, "Alt Right"},
     { DIKI_HYPER_L, "Hyper Left"},
     { DIKI_HYPER_R, "Hyper Right"},
     { DIKI_CAPS_LOCK, "Caps Lock"},
     { DIKI_NUM_LOCK, "Num Lock"},
     { DIKI_SCROLL_LOCK, "Scroll Lock"},
     { DIKI_ESCAPE, "Escape"},
     { DIKI_LEFT, "Cursor Left"},
     { DIKI_RIGHT, "Cursor Right"},
     { DIKI_UP, "Cursor Up"},
     { DIKI_DOWN, "Cursor Down"},
     { DIKI_PAGE_UP, "Page Up"},
     { DIKI_PAGE_DOWN, "Page Down"},
     { DIKI_TAB, "Tab"},
     { DIKI_ENTER, "Enter"},
     { DIKI_SPACE, "Space"},
     { DIKI_BACKSPACE, "Backspace"},
     { DIKI_INSERT, "Insert"},
     { DIKI_DELETE, "Delete"},
     { DIKI_PRINT, "Print"},
     { DIKI_PAUSE, "Pause"},
     { DIKI_QUOTE_LEFT, "Left Quote"},
     { DIKI_MINUS_SIGN, "Minus"},
     { DIKI_EQUALS_SIGN, "Equals"},
     { DIKI_BRACKET_LEFT, "Left Bracket"},
     { DIKI_BRACKET_RIGHT, "Right Bracket"},
     { DIKI_BACKSLASH, "Backslash"},
     { DIKI_SEMICOLON, "Semicolon"},
     { DIKI_QUOTE_RIGHT, "Right Quote"},
     { DIKI_COMMA, "Comma"},
     { DIKI_PERIOD, "Fullstop"},
     { DIKI_SLASH, "Slash"},
     { ((DIKT_CUSTOM) | ((27))), "A umlaut" },
     { ((DIKT_CUSTOM) | ((28))), "O umlaut" },
     { ((DIKT_CUSTOM) | ((29))), "U umlaut" },
     { ((DIKT_CUSTOM) | ((19))), "Button 1" },
     { ((DIKT_CUSTOM) | ((20))), "Button 2" },
     { DIKI_KP_0, "Key Pad 0"},
     { DIKI_KP_1, "Key Pad 1"},
     { DIKI_KP_2, "Key Pad 2"},
     { DIKI_KP_3, "Key Pad 3"},
     { DIKI_KP_4, "Key Pad 4"},
     { DIKI_KP_5, "Key Pad 5"},
     { DIKI_KP_6, "Key Pad 6"},
     { DIKI_KP_7, "Key Pad 7"},
     { DIKI_KP_8, "Key Pad 8"},
     { DIKI_KP_9, "Key Pad 9"}
};

static const int32_t keySymbolCodesSize = sizeof(keySymbolCodes)/sizeof(rcLookUp);
static const int32_t keyIdentifierCodesSize = sizeof(keyIdentifierCodes)/sizeof(rcLookUp);
# 429 "exStbKey.c"
char const * getKeyName(DFBInputEvent event)
{
    int32_t i;

    if (event.flags&DIEF_KEYID)
    {
        for(i=0; i<keyIdentifierCodesSize; i++)
        {

            if (keyIdentifierCodes[i].DFBCode == (int32_t)event.key_id)
            {
                return keyIdentifierCodes[i].name;
            }
        }
    }

    if (event.flags&DIEF_KEYSYMBOL)
    {
        for(i=0; i<keySymbolCodesSize; i++)
        {
            if (keySymbolCodes[i].DFBCode == (int32_t)event.key_symbol)
            {
                return keySymbolCodes[i].name;
            }
        }
    }

    return noName;
}


static void getRCInput( IDirectFB *dfb )
{
     DFBResult ret;
     __CPROVER_assume(dfb!=((void *)0));
     IDirectFBEventBuffer *eventBuffer;

  if (dfb != ((void *)0))
      ret = dfb->CreateInputEventBuffer(dfb, DICAPS_KEYS|DICAPS_BUTTONS|DICAPS_AXES, DFB_TRUE, &eventBuffer);

     if (ret) {
          DirectFBError( "CreateInputEventBuffer() failed", ret );
          return;
     }

     while(1)
     {
          DFBEvent event;
   if (eventBuffer!=((void *)0))
            eventBuffer->WaitForEvent(eventBuffer);

          eventBuffer->GetEvent(eventBuffer, &event);

          switch(event.input.type)
          {
              case(DIET_BUTTONPRESS) :
              case(DIET_KEYPRESS) :
                  printf("DFB Event : %s press   '%s'\n",
                         event.input.type == DIET_BUTTONPRESS ? "Button" : "Key" ,
                         getKeyName(event.input));
                  break;
              case(DIET_BUTTONRELEASE) :
              case(DIET_KEYRELEASE) :
                  printf("DFB Event : %s release '%s'\n",
                         event.input.type == DIET_BUTTONPRESS ? "Button" : "Key" ,
                         getKeyName(event.input));
                  break;
              case(DIET_AXISMOTION) :
                  printf("DFB Event : Mouse movement Direction [%2d] Speed [%2d]\n",
                         event.input.axis-DIAI_FIRST, event.input.axisrel);
                  break;
              default :
                  printf("DFB Event : Not Recognised\n");
                  break;
          }
     }
}

int main( int argc, char *argv[] )
{
    IDirectFB *dfb = ((void *)0);
    DFBResult ret;


__ESBMC_assume(argc>=0 && argc<(sizeof(argv)/sizeof(char)));
int counter;
for(counter=0; counter<argc; counter++)
  __ESBMC_assume(argv[counter]!=((void *)0));
# 543 "exStbKey.c"
    ret = DirectFBInit( &argc, &argv );
    if (ret) {
        DirectFBError( "DirectFBInit() failed", ret );
        return -1;
    }


    ret = DirectFBCreate( &dfb );
    if (ret) {
        DirectFBError( "DirectFBCreate() failed", ret );
        return -3;
    }

    getRCInput( dfb );


    if (dfb!=((void *)0))
      dfb->Release( dfb );
    return 0;
}
