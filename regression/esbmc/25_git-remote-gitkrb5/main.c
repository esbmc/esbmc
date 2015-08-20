# 1 "git-remote-gitkrb5.c"
# 1 "/home/jmorse/kerberos/client//"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "git-remote-gitkrb5.c"

# 1 "/usr/include/ctype.h" 1 3 4
# 27 "/usr/include/ctype.h" 3 4
# 1 "/usr/include/features.h" 1 3 4
# 361 "/usr/include/features.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 365 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 366 "/usr/include/sys/cdefs.h" 2 3 4
# 362 "/usr/include/features.h" 2 3 4
# 385 "/usr/include/features.h" 3 4
# 1 "/usr/include/gnu/stubs.h" 1 3 4



# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 5 "/usr/include/gnu/stubs.h" 2 3 4




# 1 "/usr/include/gnu/stubs-64.h" 1 3 4
# 10 "/usr/include/gnu/stubs.h" 2 3 4
# 386 "/usr/include/features.h" 2 3 4
# 28 "/usr/include/ctype.h" 2 3 4
# 1 "/usr/include/bits/types.h" 1 3 4
# 28 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 29 "/usr/include/bits/types.h" 2 3 4


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







typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
# 131 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/typesizes.h" 1 3 4
# 132 "/usr/include/bits/types.h" 2 3 4


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
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;

typedef int __daddr_t;
typedef long int __swblk_t;
typedef int __key_t;


typedef int __clockid_t;


typedef void * __timer_t;


typedef long int __blksize_t;




typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;


typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;


typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;

typedef long int __ssize_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;
# 29 "/usr/include/ctype.h" 2 3 4


# 41 "/usr/include/ctype.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
# 37 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/endian.h" 1 3 4
# 38 "/usr/include/endian.h" 2 3 4
# 61 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/byteswap.h" 1 3 4
# 28 "/usr/include/bits/byteswap.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 29 "/usr/include/bits/byteswap.h" 2 3 4
# 62 "/usr/include/endian.h" 2 3 4
# 42 "/usr/include/ctype.h" 2 3 4






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
# 81 "/usr/include/ctype.h" 3 4
extern __const unsigned short int **__ctype_b_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_tolower_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
extern __const __int32_t **__ctype_toupper_loc (void)
     __attribute__ ((__nothrow__)) __attribute__ ((__const));
# 96 "/usr/include/ctype.h" 3 4






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








extern int isblank (int) __attribute__ ((__nothrow__));






extern int isctype (int __c, int __mask) __attribute__ ((__nothrow__));






extern int isascii (int __c) __attribute__ ((__nothrow__));



extern int toascii (int __c) __attribute__ ((__nothrow__));



extern int _toupper (int) __attribute__ ((__nothrow__));
extern int _tolower (int) __attribute__ ((__nothrow__));
# 233 "/usr/include/ctype.h" 3 4
# 1 "/usr/include/xlocale.h" 1 3 4
# 28 "/usr/include/xlocale.h" 3 4
typedef struct __locale_struct
{

  struct __locale_data *__locales[13];


  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;


  const char *__names[13];
} *__locale_t;


typedef __locale_t locale_t;
# 234 "/usr/include/ctype.h" 2 3 4
# 247 "/usr/include/ctype.h" 3 4
extern int isalnum_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isalpha_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int iscntrl_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isdigit_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int islower_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isgraph_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isprint_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int ispunct_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isspace_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isupper_l (int, __locale_t) __attribute__ ((__nothrow__));
extern int isxdigit_l (int, __locale_t) __attribute__ ((__nothrow__));

extern int isblank_l (int, __locale_t) __attribute__ ((__nothrow__));



extern int __tolower_l (int __c, __locale_t __l) __attribute__ ((__nothrow__));
extern int tolower_l (int __c, __locale_t __l) __attribute__ ((__nothrow__));


extern int __toupper_l (int __c, __locale_t __l) __attribute__ ((__nothrow__));
extern int toupper_l (int __c, __locale_t __l) __attribute__ ((__nothrow__));
# 323 "/usr/include/ctype.h" 3 4

# 3 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/errno.h" 1 3 4
# 32 "/usr/include/errno.h" 3 4




# 1 "/usr/include/bits/errno.h" 1 3 4
# 25 "/usr/include/bits/errno.h" 3 4
# 1 "/usr/include/linux/errno.h" 1 3 4



# 1 "/usr/include/asm/errno.h" 1 3 4
# 1 "/usr/include/asm-generic/errno.h" 1 3 4



# 1 "/usr/include/asm-generic/errno-base.h" 1 3 4
# 5 "/usr/include/asm-generic/errno.h" 2 3 4
# 1 "/usr/include/asm/errno.h" 2 3 4
# 5 "/usr/include/linux/errno.h" 2 3 4
# 26 "/usr/include/bits/errno.h" 2 3 4
# 47 "/usr/include/bits/errno.h" 3 4
extern int *__errno_location (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
# 37 "/usr/include/errno.h" 2 3 4
# 55 "/usr/include/errno.h" 3 4
extern char *program_invocation_name, *program_invocation_short_name;




# 69 "/usr/include/errno.h" 3 4
typedef int error_t;
# 4 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/poll.h" 1 3 4
# 1 "/usr/include/sys/poll.h" 1 3 4
# 26 "/usr/include/sys/poll.h" 3 4
# 1 "/usr/include/bits/poll.h" 1 3 4
# 27 "/usr/include/sys/poll.h" 2 3 4


# 1 "/usr/include/bits/sigset.h" 1 3 4
# 24 "/usr/include/bits/sigset.h" 3 4
typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;
# 30 "/usr/include/sys/poll.h" 2 3 4


# 1 "/usr/include/time.h" 1 3 4
# 120 "/usr/include/time.h" 3 4
struct timespec
  {
    __time_t tv_sec;
    long int tv_nsec;
  };
# 33 "/usr/include/sys/poll.h" 2 3 4




typedef unsigned long int nfds_t;


struct pollfd
  {
    int fd;
    short int events;
    short int revents;
  };



# 58 "/usr/include/sys/poll.h" 3 4
extern int poll (struct pollfd *__fds, nfds_t __nfds, int __timeout);
# 67 "/usr/include/sys/poll.h" 3 4
extern int ppoll (struct pollfd *__fds, nfds_t __nfds,
    __const struct timespec *__timeout,
    __const __sigset_t *__ss);



# 1 "/usr/include/poll.h" 2 3 4
# 5 "git-remote-gitkrb5.c" 2
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdbool.h" 1 3 4
# 6 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/stdio.h" 1 3 4
# 30 "/usr/include/stdio.h" 3 4




# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 211 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 35 "/usr/include/stdio.h" 2 3 4
# 45 "/usr/include/stdio.h" 3 4
struct _IO_FILE;



typedef struct _IO_FILE FILE;





# 65 "/usr/include/stdio.h" 3 4
typedef struct _IO_FILE __FILE;
# 75 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/libio.h" 1 3 4
# 32 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_G_config.h" 1 3 4
# 15 "/usr/include/_G_config.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 16 "/usr/include/_G_config.h" 2 3 4




# 1 "/usr/include/wchar.h" 1 3 4
# 83 "/usr/include/wchar.h" 3 4
typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;
# 21 "/usr/include/_G_config.h" 2 3 4

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
# 53 "/usr/include/_G_config.h" 3 4
typedef int _G_int16_t __attribute__ ((__mode__ (__HI__)));
typedef int _G_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int _G_uint16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int _G_uint32_t __attribute__ ((__mode__ (__SI__)));
# 33 "/usr/include/libio.h" 2 3 4
# 53 "/usr/include/libio.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdarg.h" 1 3 4
# 40 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 54 "/usr/include/libio.h" 2 3 4
# 170 "/usr/include/libio.h" 3 4
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3 4
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
# 203 "/usr/include/libio.h" 3 4
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
# 271 "/usr/include/libio.h" 3 4
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
# 319 "/usr/include/libio.h" 3 4
  __off64_t _offset;
# 328 "/usr/include/libio.h" 3 4
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
# 364 "/usr/include/libio.h" 3 4
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, __const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);




typedef __io_read_fn cookie_read_function_t;
typedef __io_write_fn cookie_write_function_t;
typedef __io_seek_fn cookie_seek_function_t;
typedef __io_close_fn cookie_close_function_t;


typedef struct
{
  __io_read_fn *read;
  __io_write_fn *write;
  __io_seek_fn *seek;
  __io_close_fn *close;
} _IO_cookie_io_functions_t;
typedef _IO_cookie_io_functions_t cookie_io_functions_t;

struct _IO_cookie_file;


extern void _IO_cookie_init (struct _IO_cookie_file *__cfile, int __read_write,
        void *__cookie, _IO_cookie_io_functions_t __fns);







extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
# 460 "/usr/include/libio.h" 3 4
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __attribute__ ((__nothrow__));
extern int _IO_ferror (_IO_FILE *__fp) __attribute__ ((__nothrow__));

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern void _IO_funlockfile (_IO_FILE *) __attribute__ ((__nothrow__));
extern int _IO_ftrylockfile (_IO_FILE *) __attribute__ ((__nothrow__));
# 490 "/usr/include/libio.h" 3 4
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __attribute__ ((__nothrow__));
# 76 "/usr/include/stdio.h" 2 3 4




typedef __gnuc_va_list va_list;
# 91 "/usr/include/stdio.h" 3 4
typedef __off_t off_t;






typedef __off64_t off64_t;




typedef __ssize_t ssize_t;







typedef _G_fpos_t fpos_t;





typedef _G_fpos64_t fpos64_t;
# 161 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 162 "/usr/include/stdio.h" 2 3 4



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;









extern int remove (__const char *__filename) __attribute__ ((__nothrow__));

extern int rename (__const char *__old, __const char *__new) __attribute__ ((__nothrow__));




extern int renameat (int __oldfd, __const char *__old, int __newfd,
       __const char *__new) __attribute__ ((__nothrow__));








extern FILE *tmpfile (void) ;
# 204 "/usr/include/stdio.h" 3 4
extern FILE *tmpfile64 (void) ;



extern char *tmpnam (char *__s) __attribute__ ((__nothrow__)) ;





extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__)) ;
# 226 "/usr/include/stdio.h" 3 4
extern char *tempnam (__const char *__dir, __const char *__pfx)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;








extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);

# 251 "/usr/include/stdio.h" 3 4
extern int fflush_unlocked (FILE *__stream);
# 261 "/usr/include/stdio.h" 3 4
extern int fcloseall (void);









extern FILE *fopen (__const char *__restrict __filename,
      __const char *__restrict __modes) ;




extern FILE *freopen (__const char *__restrict __filename,
        __const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 294 "/usr/include/stdio.h" 3 4


extern FILE *fopen64 (__const char *__restrict __filename,
        __const char *__restrict __modes) ;
extern FILE *freopen64 (__const char *__restrict __filename,
   __const char *__restrict __modes,
   FILE *__restrict __stream) ;




extern FILE *fdopen (int __fd, __const char *__modes) __attribute__ ((__nothrow__)) ;





extern FILE *fopencookie (void *__restrict __magic_cookie,
     __const char *__restrict __modes,
     _IO_cookie_io_functions_t __io_funcs) __attribute__ ((__nothrow__)) ;




extern FILE *fmemopen (void *__s, size_t __len, __const char *__modes)
  __attribute__ ((__nothrow__)) ;




extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) __attribute__ ((__nothrow__)) ;






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






extern int vasprintf (char **__restrict __ptr, __const char *__restrict __f,
        __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 0))) ;
extern int __asprintf (char **__restrict __ptr,
         __const char *__restrict __fmt, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 3))) ;
extern int asprintf (char **__restrict __ptr,
       __const char *__restrict __fmt, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 3))) ;
# 416 "/usr/include/stdio.h" 3 4
extern int vdprintf (int __fd, __const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, __const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));








extern int fscanf (FILE *__restrict __stream,
     __const char *__restrict __format, ...) ;




extern int scanf (__const char *__restrict __format, ...) ;

extern int sscanf (__const char *__restrict __s,
     __const char *__restrict __format, ...) __attribute__ ((__nothrow__));
# 467 "/usr/include/stdio.h" 3 4








extern int vfscanf (FILE *__restrict __s, __const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;





extern int vscanf (__const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;


extern int vsscanf (__const char *__restrict __s,
      __const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__scanf__, 2, 0)));
# 526 "/usr/include/stdio.h" 3 4









extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);

# 554 "/usr/include/stdio.h" 3 4
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 565 "/usr/include/stdio.h" 3 4
extern int fgetc_unlocked (FILE *__stream);











extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);

# 598 "/usr/include/stdio.h" 3 4
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);








extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     ;






extern char *gets (char *__s) ;

# 644 "/usr/include/stdio.h" 3 4
extern char *fgets_unlocked (char *__restrict __s, int __n,
        FILE *__restrict __stream) ;
# 660 "/usr/include/stdio.h" 3 4
extern __ssize_t __getdelim (char **__restrict __lineptr,
          size_t *__restrict __n, int __delimiter,
          FILE *__restrict __stream) ;
extern __ssize_t getdelim (char **__restrict __lineptr,
        size_t *__restrict __n, int __delimiter,
        FILE *__restrict __stream) ;







extern __ssize_t getline (char **__restrict __lineptr,
       size_t *__restrict __n,
       FILE *__restrict __stream) ;








extern int fputs (__const char *__restrict __s, FILE *__restrict __stream);





extern int puts (__const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (__const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s) ;

# 721 "/usr/include/stdio.h" 3 4
extern int fputs_unlocked (__const char *__restrict __s,
      FILE *__restrict __stream);
# 732 "/usr/include/stdio.h" 3 4
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (__const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream) ;








extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);

# 768 "/usr/include/stdio.h" 3 4
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 787 "/usr/include/stdio.h" 3 4






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, __const fpos_t *__pos);
# 810 "/usr/include/stdio.h" 3 4



extern int fseeko64 (FILE *__stream, __off64_t __off, int __whence);
extern __off64_t ftello64 (FILE *__stream) ;
extern int fgetpos64 (FILE *__restrict __stream, fpos64_t *__restrict __pos);
extern int fsetpos64 (FILE *__stream, __const fpos64_t *__pos);




extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__)) ;

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;








extern void perror (__const char *__s);






# 1 "/usr/include/bits/sys_errlist.h" 1 3 4
# 27 "/usr/include/bits/sys_errlist.h" 3 4
extern int sys_nerr;
extern __const char *__const sys_errlist[];


extern int _sys_nerr;
extern __const char *__const _sys_errlist[];
# 849 "/usr/include/stdio.h" 2 3 4




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__)) ;




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__)) ;
# 868 "/usr/include/stdio.h" 3 4
extern FILE *popen (__const char *__command, __const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__));





extern char *cuserid (char *__s);




struct obstack;


extern int obstack_printf (struct obstack *__restrict __obstack,
      __const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 3)));
extern int obstack_vprintf (struct obstack *__restrict __obstack,
       __const char *__restrict __format,
       __gnuc_va_list __args)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 2, 0)));







extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__)) ;


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__));
# 938 "/usr/include/stdio.h" 3 4

# 7 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/stdlib.h" 1 3 4
# 33 "/usr/include/stdlib.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 323 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 3 4
typedef int wchar_t;
# 34 "/usr/include/stdlib.h" 2 3 4








# 1 "/usr/include/bits/waitflags.h" 1 3 4
# 43 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/bits/waitstatus.h" 1 3 4
# 67 "/usr/include/bits/waitstatus.h" 3 4
union wait
  {
    int w_status;
    struct
      {

 unsigned int __w_termsig:7;
 unsigned int __w_coredump:1;
 unsigned int __w_retcode:8;
 unsigned int:16;







      } __wait_terminated;
    struct
      {

 unsigned int __w_stopval:8;
 unsigned int __w_stopsig:8;
 unsigned int:16;






      } __wait_stopped;
  };
# 44 "/usr/include/stdlib.h" 2 3 4
# 68 "/usr/include/stdlib.h" 3 4
typedef union
  {
    union wait *__uptr;
    int *__iptr;
  } __WAIT_STATUS __attribute__ ((__transparent_union__));
# 96 "/usr/include/stdlib.h" 3 4


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


# 140 "/usr/include/stdlib.h" 3 4
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





extern float strtof (__const char *__restrict __nptr,
       char **__restrict __endptr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;

extern long double strtold (__const char *__restrict __nptr,
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

# 240 "/usr/include/stdlib.h" 3 4
extern long int strtol_l (__const char *__restrict __nptr,
     char **__restrict __endptr, int __base,
     __locale_t __loc) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 4))) ;

extern unsigned long int strtoul_l (__const char *__restrict __nptr,
        char **__restrict __endptr,
        int __base, __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 4))) ;

__extension__
extern long long int strtoll_l (__const char *__restrict __nptr,
    char **__restrict __endptr, int __base,
    __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 4))) ;

__extension__
extern unsigned long long int strtoull_l (__const char *__restrict __nptr,
       char **__restrict __endptr,
       int __base, __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 4))) ;

extern double strtod_l (__const char *__restrict __nptr,
   char **__restrict __endptr, __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3))) ;

extern float strtof_l (__const char *__restrict __nptr,
         char **__restrict __endptr, __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3))) ;

extern long double strtold_l (__const char *__restrict __nptr,
         char **__restrict __endptr,
         __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3))) ;
# 311 "/usr/include/stdlib.h" 3 4
extern char *l64a (long int __n) __attribute__ ((__nothrow__)) ;


extern long int a64l (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;




# 1 "/usr/include/sys/types.h" 1 3 4
# 28 "/usr/include/sys/types.h" 3 4






typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;




typedef __loff_t loff_t;



typedef __ino_t ino_t;






typedef __ino64_t ino64_t;




typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;
# 99 "/usr/include/sys/types.h" 3 4
typedef __pid_t pid_t;





typedef __id_t id_t;
# 116 "/usr/include/sys/types.h" 3 4
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 133 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/time.h" 1 3 4
# 58 "/usr/include/time.h" 3 4


typedef __clock_t clock_t;



# 74 "/usr/include/time.h" 3 4


typedef __time_t time_t;



# 92 "/usr/include/time.h" 3 4
typedef __clockid_t clockid_t;
# 104 "/usr/include/time.h" 3 4
typedef __timer_t timer_t;
# 134 "/usr/include/sys/types.h" 2 3 4



typedef __useconds_t useconds_t;



typedef __suseconds_t suseconds_t;





# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 148 "/usr/include/sys/types.h" 2 3 4



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
# 195 "/usr/include/sys/types.h" 3 4
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));
# 220 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/sys/select.h" 1 3 4
# 31 "/usr/include/sys/select.h" 3 4
# 1 "/usr/include/bits/select.h" 1 3 4
# 23 "/usr/include/bits/select.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 24 "/usr/include/bits/select.h" 2 3 4
# 32 "/usr/include/sys/select.h" 2 3 4


# 1 "/usr/include/bits/sigset.h" 1 3 4
# 35 "/usr/include/sys/select.h" 2 3 4



typedef __sigset_t sigset_t;





# 1 "/usr/include/time.h" 1 3 4
# 45 "/usr/include/sys/select.h" 2 3 4

# 1 "/usr/include/bits/time.h" 1 3 4
# 75 "/usr/include/bits/time.h" 3 4
struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };
# 47 "/usr/include/sys/select.h" 2 3 4
# 55 "/usr/include/sys/select.h" 3 4
typedef long int __fd_mask;
# 67 "/usr/include/sys/select.h" 3 4
typedef struct
  {



    __fd_mask fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];





  } fd_set;






typedef __fd_mask fd_mask;
# 99 "/usr/include/sys/select.h" 3 4

# 109 "/usr/include/sys/select.h" 3 4
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 121 "/usr/include/sys/select.h" 3 4
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);



# 221 "/usr/include/sys/types.h" 2 3 4


# 1 "/usr/include/sys/sysmacros.h" 1 3 4
# 30 "/usr/include/sys/sysmacros.h" 3 4
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
# 224 "/usr/include/sys/types.h" 2 3 4





typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 263 "/usr/include/sys/types.h" 3 4
typedef __blkcnt64_t blkcnt64_t;
typedef __fsblkcnt64_t fsblkcnt64_t;
typedef __fsfilcnt64_t fsfilcnt64_t;





# 1 "/usr/include/bits/pthreadtypes.h" 1 3 4
# 23 "/usr/include/bits/pthreadtypes.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 24 "/usr/include/bits/pthreadtypes.h" 2 3 4
# 50 "/usr/include/bits/pthreadtypes.h" 3 4
typedef unsigned long int pthread_t;


typedef union
{
  char __size[56];
  long int __align;
} pthread_attr_t;



typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
# 76 "/usr/include/bits/pthreadtypes.h" 3 4
typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;

    unsigned int __nusers;



    int __kind;

    int __spins;
    __pthread_list_t __list;
# 101 "/usr/include/bits/pthreadtypes.h" 3 4
  } __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;

typedef union
{
  char __size[4];
  int __align;
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
  int __align;
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
    int __writer;
    int __shared;
    unsigned long int __pad1;
    unsigned long int __pad2;


    unsigned int __flags;
  } __data;
# 187 "/usr/include/bits/pthreadtypes.h" 3 4
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
# 272 "/usr/include/sys/types.h" 2 3 4



# 321 "/usr/include/stdlib.h" 2 3 4






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



# 1 "/usr/include/alloca.h" 1 3 4
# 25 "/usr/include/alloca.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 26 "/usr/include/alloca.h" 2 3 4







extern void *alloca (size_t __size) __attribute__ ((__nothrow__));






# 498 "/usr/include/stdlib.h" 2 3 4





extern void *valloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern void abort (void) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 528 "/usr/include/stdlib.h" 3 4
extern int at_quick_exit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));







extern void quick_exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));







extern void _Exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));






extern char *getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;




extern char *__secure_getenv (__const char *__name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;





extern int putenv (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (__const char *__name, __const char *__value, int __replace)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) __attribute__ ((__nothrow__));
# 606 "/usr/include/stdlib.h" 3 4
extern char *mktemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 620 "/usr/include/stdlib.h" 3 4
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 630 "/usr/include/stdlib.h" 3 4
extern int mkstemp64 (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 642 "/usr/include/stdlib.h" 3 4
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
# 652 "/usr/include/stdlib.h" 3 4
extern int mkstemps64 (char *__template, int __suffixlen)
     __attribute__ ((__nonnull__ (1))) ;
# 663 "/usr/include/stdlib.h" 3 4
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 674 "/usr/include/stdlib.h" 3 4
extern int mkostemp (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 684 "/usr/include/stdlib.h" 3 4
extern int mkostemp64 (char *__template, int __flags) __attribute__ ((__nonnull__ (1))) ;
# 694 "/usr/include/stdlib.h" 3 4
extern int mkostemps (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;
# 706 "/usr/include/stdlib.h" 3 4
extern int mkostemps64 (char *__template, int __suffixlen, int __flags)
     __attribute__ ((__nonnull__ (1))) ;









extern int system (__const char *__command) ;






extern char *canonicalize_file_name (__const char *__name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 734 "/usr/include/stdlib.h" 3 4
extern char *realpath (__const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__)) ;






typedef int (*__compar_fn_t) (__const void *, __const void *);


typedef __compar_fn_t comparison_fn_t;



typedef int (*__compar_d_fn_t) (__const void *, __const void *, void *);





extern void *bsearch (__const void *__key, __const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;



extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));

extern void qsort_r (void *__base, size_t __nmemb, size_t __size,
       __compar_d_fn_t __compar, void *__arg)
  __attribute__ ((__nonnull__ (1, 4)));




extern int abs (int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;



__extension__ extern long long int llabs (long long int __x)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;







extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;




__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__)) ;

# 808 "/usr/include/stdlib.h" 3 4
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
# 896 "/usr/include/stdlib.h" 3 4
extern int getsubopt (char **__restrict __optionp,
        char *__const *__restrict __tokens,
        char **__restrict __valuep)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2, 3))) ;





extern void setkey (__const char *__key) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));







extern int posix_openpt (int __oflag) ;







extern int grantpt (int __fd) __attribute__ ((__nothrow__));



extern int unlockpt (int __fd) __attribute__ ((__nothrow__));




extern char *ptsname (int __fd) __attribute__ ((__nothrow__)) ;






extern int ptsname_r (int __fd, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));


extern int getpt (void);






extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 964 "/usr/include/stdlib.h" 3 4

# 8 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/string.h" 1 3 4
# 29 "/usr/include/string.h" 3 4





# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 35 "/usr/include/string.h" 2 3 4









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
# 95 "/usr/include/string.h" 3 4
extern void *memchr (__const void *__s, int __c, size_t __n)
      __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


# 109 "/usr/include/string.h" 3 4
extern void *rawmemchr (__const void *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 120 "/usr/include/string.h" 3 4
extern void *memrchr (__const void *__s, int __c, size_t __n)
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

# 165 "/usr/include/string.h" 3 4
extern int strcoll_l (__const char *__s1, __const char *__s2, __locale_t __l)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));

extern size_t strxfrm_l (char *__dest, __const char *__src, size_t __n,
    __locale_t __l) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));





extern char *strdup (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));






extern char *strndup (__const char *__string, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__malloc__)) __attribute__ ((__nonnull__ (1)));
# 210 "/usr/include/string.h" 3 4

# 235 "/usr/include/string.h" 3 4
extern char *strchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 262 "/usr/include/string.h" 3 4
extern char *strrchr (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));


# 276 "/usr/include/string.h" 3 4
extern char *strchrnul (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));






extern size_t strcspn (__const char *__s, __const char *__reject)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern size_t strspn (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 314 "/usr/include/string.h" 3 4
extern char *strpbrk (__const char *__s, __const char *__accept)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 342 "/usr/include/string.h" 3 4
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
# 373 "/usr/include/string.h" 3 4
extern char *strcasestr (__const char *__haystack, __const char *__needle)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));







extern void *memmem (__const void *__haystack, size_t __haystacklen,
       __const void *__needle, size_t __needlelen)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 3)));



extern void *__mempcpy (void *__restrict __dest,
   __const void *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern void *mempcpy (void *__restrict __dest,
        __const void *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));





extern size_t strlen (__const char *__s)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));





extern size_t strnlen (__const char *__string, size_t __maxlen)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));





extern char *strerror (int __errnum) __attribute__ ((__nothrow__));

# 438 "/usr/include/string.h" 3 4
extern char *strerror_r (int __errnum, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));





extern char *strerror_l (int __errnum, __locale_t __l) __attribute__ ((__nothrow__));





extern void __bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern void bcopy (__const void *__src, void *__dest, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));


extern void bzero (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int bcmp (__const void *__s1, __const void *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));
# 489 "/usr/include/string.h" 3 4
extern char *index (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));
# 517 "/usr/include/string.h" 3 4
extern char *rindex (__const char *__s, int __c)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));




extern int ffs (int __i) __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int ffsl (long int __l) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

__extension__ extern int ffsll (long long int __ll)
     __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int strcasecmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern int strncasecmp (__const char *__s1, __const char *__s2, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));





extern int strcasecmp_l (__const char *__s1, __const char *__s2,
    __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 3)));

extern int strncasecmp_l (__const char *__s1, __const char *__s2,
     size_t __n, __locale_t __loc)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2, 4)));





extern char *strsep (char **__restrict __stringp,
       __const char *__restrict __delim)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern char *strsignal (int __sig) __attribute__ ((__nothrow__));


extern char *__stpcpy (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern char *stpcpy (char *__restrict __dest, __const char *__restrict __src)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern char *__stpncpy (char *__restrict __dest,
   __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern char *stpncpy (char *__restrict __dest,
        __const char *__restrict __src, size_t __n)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int strverscmp (__const char *__s1, __const char *__s2)
     __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1, 2)));


extern char *strfry (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern void *memfrob (void *__s, size_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 606 "/usr/include/string.h" 3 4
extern char *basename (__const char *__filename) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 646 "/usr/include/string.h" 3 4

# 9 "git-remote-gitkrb5.c" 2
# 1 "/usr/include/unistd.h" 1 3 4
# 28 "/usr/include/unistd.h" 3 4

# 203 "/usr/include/unistd.h" 3 4
# 1 "/usr/include/bits/posix_opt.h" 1 3 4
# 204 "/usr/include/unistd.h" 2 3 4



# 1 "/usr/include/bits/environments.h" 1 3 4
# 23 "/usr/include/bits/environments.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 24 "/usr/include/bits/environments.h" 2 3 4
# 208 "/usr/include/unistd.h" 2 3 4
# 227 "/usr/include/unistd.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 228 "/usr/include/unistd.h" 2 3 4
# 268 "/usr/include/unistd.h" 3 4
typedef __intptr_t intptr_t;






typedef __socklen_t socklen_t;
# 288 "/usr/include/unistd.h" 3 4
extern int access (__const char *__name, int __type) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));




extern int euidaccess (__const char *__name, int __type)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern int eaccess (__const char *__name, int __type)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern int faccessat (int __fd, __const char *__file, int __type, int __flag)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;
# 331 "/usr/include/unistd.h" 3 4
extern __off_t lseek (int __fd, __off_t __offset, int __whence) __attribute__ ((__nothrow__));
# 342 "/usr/include/unistd.h" 3 4
extern __off64_t lseek64 (int __fd, __off64_t __offset, int __whence)
     __attribute__ ((__nothrow__));






extern int close (int __fd);






extern ssize_t read (int __fd, void *__buf, size_t __nbytes) ;





extern ssize_t write (int __fd, __const void *__buf, size_t __n) ;
# 373 "/usr/include/unistd.h" 3 4
extern ssize_t pread (int __fd, void *__buf, size_t __nbytes,
        __off_t __offset) ;






extern ssize_t pwrite (int __fd, __const void *__buf, size_t __n,
         __off_t __offset) ;
# 401 "/usr/include/unistd.h" 3 4
extern ssize_t pread64 (int __fd, void *__buf, size_t __nbytes,
   __off64_t __offset) ;


extern ssize_t pwrite64 (int __fd, __const void *__buf, size_t __n,
    __off64_t __offset) ;







extern int pipe (int __pipedes[2]) __attribute__ ((__nothrow__)) ;




extern int pipe2 (int __pipedes[2], int __flags) __attribute__ ((__nothrow__)) ;
# 429 "/usr/include/unistd.h" 3 4
extern unsigned int alarm (unsigned int __seconds) __attribute__ ((__nothrow__));
# 441 "/usr/include/unistd.h" 3 4
extern unsigned int sleep (unsigned int __seconds);







extern __useconds_t ualarm (__useconds_t __value, __useconds_t __interval)
     __attribute__ ((__nothrow__));






extern int usleep (__useconds_t __useconds);
# 466 "/usr/include/unistd.h" 3 4
extern int pause (void);



extern int chown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchown (int __fd, __uid_t __owner, __gid_t __group) __attribute__ ((__nothrow__)) ;




extern int lchown (__const char *__file, __uid_t __owner, __gid_t __group)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;






extern int fchownat (int __fd, __const char *__file, __uid_t __owner,
       __gid_t __group, int __flag)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;



extern int chdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;



extern int fchdir (int __fd) __attribute__ ((__nothrow__)) ;
# 508 "/usr/include/unistd.h" 3 4
extern char *getcwd (char *__buf, size_t __size) __attribute__ ((__nothrow__)) ;





extern char *get_current_dir_name (void) __attribute__ ((__nothrow__));







extern char *getwd (char *__buf)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__deprecated__)) ;




extern int dup (int __fd) __attribute__ ((__nothrow__)) ;


extern int dup2 (int __fd, int __fd2) __attribute__ ((__nothrow__));




extern int dup3 (int __fd, int __fd2, int __flags) __attribute__ ((__nothrow__));



extern char **__environ;

extern char **environ;





extern int execve (__const char *__path, char *__const __argv[],
     char *__const __envp[]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int fexecve (int __fd, char *__const __argv[], char *__const __envp[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));




extern int execv (__const char *__path, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int execle (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int execl (__const char *__path, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern int execvp (__const char *__file, char *__const __argv[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int execlp (__const char *__file, __const char *__arg, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));




extern int execvpe (__const char *__file, char *__const __argv[],
      char *__const __envp[])
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));





extern int nice (int __inc) __attribute__ ((__nothrow__)) ;




extern void _exit (int __status) __attribute__ ((__noreturn__));





# 1 "/usr/include/bits/confname.h" 1 3 4
# 26 "/usr/include/bits/confname.h" 3 4
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

    _SC_RAW_SOCKETS,


    _SC_V7_ILP32_OFF32,

    _SC_V7_ILP32_OFFBIG,

    _SC_V7_LP64_OFF64,

    _SC_V7_LPBIG_OFFBIG,


    _SC_SS_REPL_MAX,


    _SC_TRACE_EVENT_NAME_MAX,

    _SC_TRACE_NAME_MAX,

    _SC_TRACE_SYS_MAX,

    _SC_TRACE_USER_EVENT_MAX,


    _SC_XOPEN_STREAMS,


    _SC_THREAD_ROBUST_PRIO_INHERIT,

    _SC_THREAD_ROBUST_PRIO_PROTECT

  };


enum
  {
    _CS_PATH,


    _CS_V6_WIDTH_RESTRICTED_ENVS,



    _CS_GNU_LIBC_VERSION,

    _CS_GNU_LIBPTHREAD_VERSION,


    _CS_V5_WIDTH_RESTRICTED_ENVS,



    _CS_V7_WIDTH_RESTRICTED_ENVS,



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

    _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS,


    _CS_POSIX_V7_ILP32_OFF32_CFLAGS,

    _CS_POSIX_V7_ILP32_OFF32_LDFLAGS,

    _CS_POSIX_V7_ILP32_OFF32_LIBS,

    _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS,

    _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS,

    _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS,

    _CS_POSIX_V7_ILP32_OFFBIG_LIBS,

    _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS,

    _CS_POSIX_V7_LP64_OFF64_CFLAGS,

    _CS_POSIX_V7_LP64_OFF64_LDFLAGS,

    _CS_POSIX_V7_LP64_OFF64_LIBS,

    _CS_POSIX_V7_LP64_OFF64_LINTFLAGS,

    _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS,

    _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS,

    _CS_POSIX_V7_LPBIG_OFFBIG_LIBS,

    _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS,


    _CS_V6_ENV,

    _CS_V7_ENV

  };
# 607 "/usr/include/unistd.h" 2 3 4


extern long int pathconf (__const char *__path, int __name)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));


extern long int fpathconf (int __fd, int __name) __attribute__ ((__nothrow__));


extern long int sysconf (int __name) __attribute__ ((__nothrow__));



extern size_t confstr (int __name, char *__buf, size_t __len) __attribute__ ((__nothrow__));




extern __pid_t getpid (void) __attribute__ ((__nothrow__));


extern __pid_t getppid (void) __attribute__ ((__nothrow__));




extern __pid_t getpgrp (void) __attribute__ ((__nothrow__));
# 643 "/usr/include/unistd.h" 3 4
extern __pid_t __getpgid (__pid_t __pid) __attribute__ ((__nothrow__));

extern __pid_t getpgid (__pid_t __pid) __attribute__ ((__nothrow__));






extern int setpgid (__pid_t __pid, __pid_t __pgid) __attribute__ ((__nothrow__));
# 669 "/usr/include/unistd.h" 3 4
extern int setpgrp (void) __attribute__ ((__nothrow__));
# 686 "/usr/include/unistd.h" 3 4
extern __pid_t setsid (void) __attribute__ ((__nothrow__));



extern __pid_t getsid (__pid_t __pid) __attribute__ ((__nothrow__));



extern __uid_t getuid (void) __attribute__ ((__nothrow__));


extern __uid_t geteuid (void) __attribute__ ((__nothrow__));


extern __gid_t getgid (void) __attribute__ ((__nothrow__));


extern __gid_t getegid (void) __attribute__ ((__nothrow__));




extern int getgroups (int __size, __gid_t __list[]) __attribute__ ((__nothrow__)) ;



extern int group_member (__gid_t __gid) __attribute__ ((__nothrow__));






extern int setuid (__uid_t __uid) __attribute__ ((__nothrow__));




extern int setreuid (__uid_t __ruid, __uid_t __euid) __attribute__ ((__nothrow__));




extern int seteuid (__uid_t __uid) __attribute__ ((__nothrow__));






extern int setgid (__gid_t __gid) __attribute__ ((__nothrow__));




extern int setregid (__gid_t __rgid, __gid_t __egid) __attribute__ ((__nothrow__));




extern int setegid (__gid_t __gid) __attribute__ ((__nothrow__));





extern int getresuid (__uid_t *__ruid, __uid_t *__euid, __uid_t *__suid)
     __attribute__ ((__nothrow__));



extern int getresgid (__gid_t *__rgid, __gid_t *__egid, __gid_t *__sgid)
     __attribute__ ((__nothrow__));



extern int setresuid (__uid_t __ruid, __uid_t __euid, __uid_t __suid)
     __attribute__ ((__nothrow__));



extern int setresgid (__gid_t __rgid, __gid_t __egid, __gid_t __sgid)
     __attribute__ ((__nothrow__));






extern __pid_t fork (void) __attribute__ ((__nothrow__));







extern __pid_t vfork (void) __attribute__ ((__nothrow__));





extern char *ttyname (int __fd) __attribute__ ((__nothrow__));



extern int ttyname_r (int __fd, char *__buf, size_t __buflen)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2))) ;



extern int isatty (int __fd) __attribute__ ((__nothrow__));





extern int ttyslot (void) __attribute__ ((__nothrow__));




extern int link (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern int linkat (int __fromfd, __const char *__from, int __tofd,
     __const char *__to, int __flags)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4))) ;




extern int symlink (__const char *__from, __const char *__to)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern ssize_t readlink (__const char *__restrict __path,
    char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2))) ;




extern int symlinkat (__const char *__from, int __tofd,
        __const char *__to) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3))) ;


extern ssize_t readlinkat (int __fd, __const char *__restrict __path,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 3))) ;



extern int unlink (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern int unlinkat (int __fd, __const char *__name, int __flag)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));



extern int rmdir (__const char *__path) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));



extern __pid_t tcgetpgrp (int __fd) __attribute__ ((__nothrow__));


extern int tcsetpgrp (int __fd, __pid_t __pgrp_id) __attribute__ ((__nothrow__));






extern char *getlogin (void);







extern int getlogin_r (char *__name, size_t __name_len) __attribute__ ((__nonnull__ (1)));




extern int setlogin (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
# 890 "/usr/include/unistd.h" 3 4
# 1 "/usr/include/getopt.h" 1 3 4
# 59 "/usr/include/getopt.h" 3 4
extern char *optarg;
# 73 "/usr/include/getopt.h" 3 4
extern int optind;




extern int opterr;



extern int optopt;
# 152 "/usr/include/getopt.h" 3 4
extern int getopt (int ___argc, char *const *___argv, const char *__shortopts)
       __attribute__ ((__nothrow__));
# 891 "/usr/include/unistd.h" 2 3 4







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
# 976 "/usr/include/unistd.h" 3 4
extern int fsync (int __fd);






extern long int gethostid (void);


extern void sync (void) __attribute__ ((__nothrow__));





extern int getpagesize (void) __attribute__ ((__nothrow__)) __attribute__ ((__const__));




extern int getdtablesize (void) __attribute__ ((__nothrow__));
# 1007 "/usr/include/unistd.h" 3 4
extern int truncate (__const char *__file, __off_t __length)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 1019 "/usr/include/unistd.h" 3 4
extern int truncate64 (__const char *__file, __off64_t __length)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1))) ;
# 1029 "/usr/include/unistd.h" 3 4
extern int ftruncate (int __fd, __off_t __length) __attribute__ ((__nothrow__)) ;
# 1039 "/usr/include/unistd.h" 3 4
extern int ftruncate64 (int __fd, __off64_t __length) __attribute__ ((__nothrow__)) ;
# 1050 "/usr/include/unistd.h" 3 4
extern int brk (void *__addr) __attribute__ ((__nothrow__)) ;





extern void *sbrk (intptr_t __delta) __attribute__ ((__nothrow__));
# 1071 "/usr/include/unistd.h" 3 4
extern long int syscall (long int __sysno, ...) __attribute__ ((__nothrow__));
# 1094 "/usr/include/unistd.h" 3 4
extern int lockf (int __fd, int __cmd, __off_t __len) ;
# 1104 "/usr/include/unistd.h" 3 4
extern int lockf64 (int __fd, int __cmd, __off64_t __len) ;
# 1125 "/usr/include/unistd.h" 3 4
extern int fdatasync (int __fildes);







extern char *crypt (__const char *__key, __const char *__salt)
     __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));



extern void encrypt (char *__block, int __edflag) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));






extern void swab (__const void *__restrict __from, void *__restrict __to,
    ssize_t __n) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));







extern char *ctermid (char *__s) __attribute__ ((__nothrow__));
# 1163 "/usr/include/unistd.h" 3 4

# 10 "git-remote-gitkrb5.c" 2

# 1 "/usr/include/krb5/krb5.h" 1 3 4
# 102 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 1 3 4
# 34 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/syslimits.h" 1 3 4






# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 1 3 4
# 169 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 3 4
# 1 "/usr/include/limits.h" 1 3 4
# 145 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/posix1_lim.h" 1 3 4
# 157 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/local_lim.h" 1 3 4
# 39 "/usr/include/bits/local_lim.h" 3 4
# 1 "/usr/include/linux/limits.h" 1 3 4
# 40 "/usr/include/bits/local_lim.h" 2 3 4
# 158 "/usr/include/bits/posix1_lim.h" 2 3 4
# 146 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/posix2_lim.h" 1 3 4
# 150 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/xopen_lim.h" 1 3 4
# 34 "/usr/include/bits/xopen_lim.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 35 "/usr/include/bits/xopen_lim.h" 2 3 4
# 154 "/usr/include/limits.h" 2 3 4
# 170 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 2 3 4
# 8 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/syslimits.h" 2 3 4
# 35 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/limits.h" 2 3 4
# 103 "/usr/include/krb5/krb5.h" 2 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdarg.h" 1 3 4
# 104 "/usr/include/krb5/krb5.h" 2 3 4
# 115 "/usr/include/krb5/krb5.h" 3 4

# 130 "/usr/include/krb5/krb5.h" 3 4
struct _profile_t;
# 141 "/usr/include/krb5/krb5.h" 3 4
typedef unsigned char krb5_octet;





typedef short krb5_int16;
typedef unsigned short krb5_ui_2;





typedef int krb5_int32;
typedef unsigned int krb5_ui_4;
# 198 "/usr/include/krb5/krb5.h" 3 4
typedef unsigned int krb5_boolean;
typedef unsigned int krb5_msgtype;
typedef unsigned int krb5_kvno;

typedef krb5_int32 krb5_addrtype;
typedef krb5_int32 krb5_enctype;
typedef krb5_int32 krb5_cksumtype;
typedef krb5_int32 krb5_authdatatype;
typedef krb5_int32 krb5_keyusage;
typedef krb5_int32 krb5_cryptotype;

typedef krb5_int32 krb5_preauthtype;
typedef krb5_int32 krb5_flags;
typedef krb5_int32 krb5_timestamp;
typedef krb5_int32 krb5_error_code;
typedef krb5_int32 krb5_deltat;

typedef krb5_error_code krb5_magic;

typedef struct _krb5_data {
    krb5_magic magic;
    unsigned int length;
    char *data;
} krb5_data;

typedef struct _krb5_octet_data {
    krb5_magic magic;
    unsigned int length;
    krb5_octet *data;
} krb5_octet_data;
# 238 "/usr/include/krb5/krb5.h" 3 4
typedef void * krb5_pointer;
typedef void const * krb5_const_pointer;

typedef struct krb5_principal_data {
    krb5_magic magic;
    krb5_data realm;
    krb5_data *data;
    krb5_int32 length;
    krb5_int32 type;
} krb5_principal_data;

typedef krb5_principal_data * krb5_principal;
# 284 "/usr/include/krb5/krb5.h" 3 4
typedef const krb5_principal_data *krb5_const_principal;
# 306 "/usr/include/krb5/krb5.h" 3 4
krb5_boolean krb5_is_referral_realm(const krb5_data *);


const krb5_data * krb5_anonymous_realm(void);
krb5_const_principal krb5_anonymous_principal(void);
# 323 "/usr/include/krb5/krb5.h" 3 4
typedef struct _krb5_address {
    krb5_magic magic;
    krb5_addrtype addrtype;
    unsigned int length;
    krb5_octet *contents;
} krb5_address;
# 350 "/usr/include/krb5/krb5.h" 3 4
struct _krb5_context;
typedef struct _krb5_context * krb5_context;

struct _krb5_auth_context;
typedef struct _krb5_auth_context * krb5_auth_context;

struct _krb5_cryptosystem_entry;






typedef struct _krb5_keyblock {
    krb5_magic magic;
    krb5_enctype enctype;
    unsigned int length;
    krb5_octet *contents;
} krb5_keyblock;
# 377 "/usr/include/krb5/krb5.h" 3 4
struct krb5_key_st;
typedef struct krb5_key_st *krb5_key;


typedef struct _krb5_encrypt_block {
    krb5_magic magic;
    krb5_enctype crypto_entry;


    krb5_keyblock *key;
} krb5_encrypt_block;


typedef struct _krb5_checksum {
    krb5_magic magic;
    krb5_cksumtype checksum_type;
    unsigned int length;
    krb5_octet *contents;
} krb5_checksum;

typedef struct _krb5_enc_data {
    krb5_magic magic;
    krb5_enctype enctype;
    krb5_kvno kvno;
    krb5_data ciphertext;
} krb5_enc_data;

typedef struct _krb5_crypto_iov {
    krb5_cryptotype flags;
    krb5_data data;
} krb5_crypto_iov;
# 458 "/usr/include/krb5/krb5.h" 3 4
enum {
    KRB5_C_RANDSOURCE_OLDAPI = 0,
    KRB5_C_RANDSOURCE_OSRAND = 1,
    KRB5_C_RANDSOURCE_TRUSTEDPARTY = 2,




    KRB5_C_RANDSOURCE_TIMING = 3,
    KRB5_C_RANDSOURCE_EXTERNAL_PROTOCOL = 4,
    KRB5_C_RANDSOURCE_MAX = 5
};
# 486 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_c_encrypt(krb5_context context, const krb5_keyblock *key,
               krb5_keyusage usage, const krb5_data *cipher_state,
               const krb5_data *input, krb5_enc_data *output);

krb5_error_code
krb5_c_decrypt(krb5_context context, const krb5_keyblock *key,
               krb5_keyusage usage, const krb5_data *cipher_state,
               const krb5_enc_data *input, krb5_data *output);

krb5_error_code
krb5_c_encrypt_length(krb5_context context, krb5_enctype enctype,
                      size_t inputlen, size_t *length);

krb5_error_code
krb5_c_block_size(krb5_context context, krb5_enctype enctype,
                  size_t *blocksize);

krb5_error_code
krb5_c_keylengths(krb5_context context, krb5_enctype enctype,
                  size_t *keybytes, size_t *keylength);

krb5_error_code
krb5_c_init_state(krb5_context context, const krb5_keyblock *key,
                  krb5_keyusage usage, krb5_data *new_state);

krb5_error_code
krb5_c_free_state(krb5_context context, const krb5_keyblock *key,
                  krb5_data *state);

krb5_error_code
krb5_c_prf(krb5_context, const krb5_keyblock *, krb5_data *in, krb5_data *out);

krb5_error_code
krb5_c_prf_length(krb5_context, krb5_enctype, size_t *outlen);

krb5_error_code
krb5_c_fx_cf2_simple(krb5_context context,
                     krb5_keyblock *k1, const char *pepper1,
                     krb5_keyblock *k2, const char *pepper2,
                     krb5_keyblock **out);
# 537 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_c_make_random_key(krb5_context context, krb5_enctype enctype,
                       krb5_keyblock *k5_random_key);

krb5_error_code
krb5_c_random_to_key(krb5_context context, krb5_enctype enctype,
                     krb5_data *random_data, krb5_keyblock *k5_random_key);






krb5_error_code
krb5_c_random_add_entropy(krb5_context context, unsigned int randsource_id,
                          const krb5_data *data);

krb5_error_code
krb5_c_random_make_octets(krb5_context context, krb5_data *data);
# 566 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_c_random_os_entropy(krb5_context context, int strong, int *success);

               krb5_error_code
krb5_c_random_seed(krb5_context context, krb5_data *data);

krb5_error_code
krb5_c_string_to_key(krb5_context context, krb5_enctype enctype,
                     const krb5_data *string, const krb5_data *salt,
                     krb5_keyblock *key);

krb5_error_code
krb5_c_string_to_key_with_params(krb5_context context,
                                 krb5_enctype enctype,
                                 const krb5_data *string,
                                 const krb5_data *salt,
                                 const krb5_data *params,
                                 krb5_keyblock *key);

krb5_error_code
krb5_c_enctype_compare(krb5_context context, krb5_enctype e1, krb5_enctype e2,
                       krb5_boolean *similar);

krb5_error_code
krb5_c_make_checksum(krb5_context context, krb5_cksumtype cksumtype,
                     const krb5_keyblock *key, krb5_keyusage usage,
                     const krb5_data *input, krb5_checksum *cksum);

krb5_error_code
krb5_c_verify_checksum(krb5_context context, const krb5_keyblock *key,
                       krb5_keyusage usage, const krb5_data *data,
                       const krb5_checksum *cksum, krb5_boolean *valid);

krb5_error_code
krb5_c_checksum_length(krb5_context context, krb5_cksumtype cksumtype,
                       size_t *length);

krb5_error_code
krb5_c_keyed_checksum_types(krb5_context context, krb5_enctype enctype,
                            unsigned int *count, krb5_cksumtype **cksumtypes);
# 659 "/usr/include/krb5/krb5.h" 3 4
krb5_boolean krb5_c_valid_enctype(krb5_enctype ktype);
krb5_boolean krb5_c_valid_cksumtype(krb5_cksumtype ctype);
krb5_boolean krb5_c_is_coll_proof_cksum(krb5_cksumtype ctype);
krb5_boolean krb5_c_is_keyed_cksum(krb5_cksumtype ctype);
# 674 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_c_make_checksum_iov(krb5_context context, krb5_cksumtype cksumtype,
                         const krb5_keyblock *key, krb5_keyusage usage,
                         krb5_crypto_iov *data, size_t num_data);

krb5_error_code
krb5_c_verify_checksum_iov(krb5_context context, krb5_cksumtype cksumtype,
                           const krb5_keyblock *key, krb5_keyusage usage,
                           const krb5_crypto_iov *data, size_t num_data,
                           krb5_boolean *valid);

krb5_error_code
krb5_c_encrypt_iov(krb5_context context, const krb5_keyblock *key,
                   krb5_keyusage usage, const krb5_data *cipher_state,
                   krb5_crypto_iov *data, size_t num_data);

krb5_error_code
krb5_c_decrypt_iov(krb5_context context, const krb5_keyblock *key,
                   krb5_keyusage usage, const krb5_data *cipher_state,
                   krb5_crypto_iov *data, size_t num_data);

krb5_error_code
krb5_c_crypto_length(krb5_context context, krb5_enctype enctype,
                     krb5_cryptotype type, unsigned int *size);

krb5_error_code
krb5_c_crypto_length_iov(krb5_context context, krb5_enctype enctype,
                         krb5_crypto_iov *data, size_t num_data);

krb5_error_code
krb5_c_padding_length(krb5_context context, krb5_enctype enctype,
                      size_t data_length, unsigned int *size);

krb5_error_code
krb5_k_create_key(krb5_context context, const krb5_keyblock *key_data,
                  krb5_key *out);


void krb5_k_reference_key(krb5_context context, krb5_key key);


void krb5_k_free_key(krb5_context context, krb5_key key);

krb5_error_code
krb5_k_key_keyblock(krb5_context context, krb5_key key,
                    krb5_keyblock **key_data);

krb5_enctype
krb5_k_key_enctype(krb5_context context, krb5_key key);

krb5_error_code
krb5_k_encrypt(krb5_context context, krb5_key key, krb5_keyusage usage,
               const krb5_data *cipher_state, const krb5_data *input,
               krb5_enc_data *output);

krb5_error_code
krb5_k_encrypt_iov(krb5_context context, krb5_key key, krb5_keyusage usage,
                   const krb5_data *cipher_state, krb5_crypto_iov *data,
                   size_t num_data);

krb5_error_code
krb5_k_decrypt(krb5_context context, krb5_key key, krb5_keyusage usage,
               const krb5_data *cipher_state, const krb5_enc_data *input,
               krb5_data *output);

krb5_error_code
krb5_k_decrypt_iov(krb5_context context, krb5_key key, krb5_keyusage usage,
                   const krb5_data *cipher_state, krb5_crypto_iov *data,
                   size_t num_data);

krb5_error_code
krb5_k_make_checksum(krb5_context context, krb5_cksumtype cksumtype,
                     krb5_key key, krb5_keyusage usage, const krb5_data *input,
                     krb5_checksum *cksum);

krb5_error_code
krb5_k_make_checksum_iov(krb5_context context, krb5_cksumtype cksumtype,
                         krb5_key key, krb5_keyusage usage,
                         krb5_crypto_iov *data, size_t num_data);

krb5_error_code
krb5_k_verify_checksum(krb5_context context, krb5_key key, krb5_keyusage usage,
                       const krb5_data *data, const krb5_checksum *cksum,
                       krb5_boolean *valid);

krb5_error_code
krb5_k_verify_checksum_iov(krb5_context context, krb5_cksumtype cksumtype,
                           krb5_key key, krb5_keyusage usage,
                           const krb5_crypto_iov *data, size_t num_data,
                           krb5_boolean *valid);

krb5_error_code
krb5_k_prf(krb5_context context, krb5_key key, krb5_data *in, krb5_data *out);






krb5_error_code
krb5_encrypt(krb5_context context, krb5_const_pointer inptr,
             krb5_pointer outptr, size_t size, krb5_encrypt_block *eblock,
             krb5_pointer ivec);

krb5_error_code
krb5_decrypt(krb5_context context, krb5_const_pointer inptr,
             krb5_pointer outptr, size_t size, krb5_encrypt_block *eblock,
             krb5_pointer ivec);

krb5_error_code
krb5_process_key(krb5_context context, krb5_encrypt_block *eblock,
                 const krb5_keyblock * key);

krb5_error_code
krb5_finish_key(krb5_context context, krb5_encrypt_block * eblock);

krb5_error_code
krb5_string_to_key(krb5_context context, const krb5_encrypt_block *eblock,
                   krb5_keyblock * keyblock, const krb5_data *data,
                   const krb5_data *salt);

krb5_error_code
krb5_init_random_key(krb5_context context, const krb5_encrypt_block *eblock,
                     const krb5_keyblock *keyblock, krb5_pointer *ptr);

krb5_error_code
krb5_finish_random_key(krb5_context context, const krb5_encrypt_block *eblock,
                       krb5_pointer *ptr);

krb5_error_code
krb5_random_key(krb5_context context, const krb5_encrypt_block *eblock,
                krb5_pointer ptr, krb5_keyblock **keyblock);

krb5_enctype
krb5_eblock_enctype(krb5_context context, const krb5_encrypt_block *eblock);

krb5_error_code
krb5_use_enctype(krb5_context context, krb5_encrypt_block *eblock,
                 krb5_enctype enctype);

size_t
krb5_encrypt_size(size_t length, krb5_enctype crypto);

size_t
krb5_checksum_size(krb5_context context, krb5_cksumtype ctype);

krb5_error_code
krb5_calculate_checksum(krb5_context context, krb5_cksumtype ctype,
                        krb5_const_pointer in, size_t in_length,
                        krb5_const_pointer seed, size_t seed_length,
                        krb5_checksum * outcksum);

krb5_error_code
krb5_verify_checksum(krb5_context context, krb5_cksumtype ctype,
                     const krb5_checksum * cksum, krb5_const_pointer in,
                     size_t in_length, krb5_const_pointer seed,
                     size_t seed_length);
# 1098 "/usr/include/krb5/krb5.h" 3 4
typedef struct _krb5_ticket_times {
    krb5_timestamp authtime;

    krb5_timestamp starttime;

    krb5_timestamp endtime;
    krb5_timestamp renew_till;
} krb5_ticket_times;


typedef struct _krb5_authdata {
    krb5_magic magic;
    krb5_authdatatype ad_type;
    unsigned int length;
    krb5_octet *contents;
} krb5_authdata;


typedef struct _krb5_transited {
    krb5_magic magic;
    krb5_octet tr_type;
    krb5_data tr_contents;
} krb5_transited;

typedef struct _krb5_enc_tkt_part {
    krb5_magic magic;

    krb5_flags flags;
    krb5_keyblock *session;
    krb5_principal client;
    krb5_transited transited;
    krb5_ticket_times times;
    krb5_address **caddrs;
    krb5_authdata **authorization_data;
} krb5_enc_tkt_part;

typedef struct _krb5_ticket {
    krb5_magic magic;

    krb5_principal server;
    krb5_enc_data enc_part;

    krb5_enc_tkt_part *enc_part2;

} krb5_ticket;


typedef struct _krb5_authenticator {
    krb5_magic magic;
    krb5_principal client;
    krb5_checksum *checksum;
    krb5_int32 cusec;
    krb5_timestamp ctime;
    krb5_keyblock *subkey;
    krb5_ui_4 seq_number;
    krb5_authdata **authorization_data;
} krb5_authenticator;

typedef struct _krb5_tkt_authent {
    krb5_magic magic;
    krb5_ticket *ticket;
    krb5_authenticator *authenticator;
    krb5_flags ap_options;
} krb5_tkt_authent;


typedef struct _krb5_creds {
    krb5_magic magic;
    krb5_principal client;
    krb5_principal server;
    krb5_keyblock keyblock;
    krb5_ticket_times times;
    krb5_boolean is_skey;

    krb5_flags ticket_flags;
    krb5_address **addresses;
    krb5_data ticket;
    krb5_data second_ticket;


    krb5_authdata **authdata;
} krb5_creds;


typedef struct _krb5_last_req_entry {
    krb5_magic magic;
    krb5_int32 lr_type;
    krb5_timestamp value;
} krb5_last_req_entry;


typedef struct _krb5_pa_data {
    krb5_magic magic;
    krb5_preauthtype pa_type;
    unsigned int length;
    krb5_octet *contents;
} krb5_pa_data;







typedef struct _krb5_typed_data {
    krb5_magic magic;
    krb5_int32 type;
    unsigned int length;
    krb5_octet *data;
} krb5_typed_data;

typedef struct _krb5_kdc_req {
    krb5_magic magic;
    krb5_msgtype msg_type;
    krb5_pa_data **padata;

    krb5_flags kdc_options;
    krb5_principal client;
    krb5_principal server;

    krb5_timestamp from;
    krb5_timestamp till;
    krb5_timestamp rtime;
    krb5_int32 nonce;
    int nktypes;
    krb5_enctype *ktype;
    krb5_address **addresses;
    krb5_enc_data authorization_data;
    krb5_authdata **unenc_authdata;

    krb5_ticket **second_ticket;






    void * kdc_state;
} krb5_kdc_req;

typedef struct _krb5_enc_kdc_rep_part {
    krb5_magic magic;

    krb5_msgtype msg_type;
    krb5_keyblock *session;
    krb5_last_req_entry **last_req;
    krb5_int32 nonce;
    krb5_timestamp key_exp;
    krb5_flags flags;
    krb5_ticket_times times;
    krb5_principal server;
    krb5_address **caddrs;

    krb5_pa_data **enc_padata;
} krb5_enc_kdc_rep_part;

typedef struct _krb5_kdc_rep {
    krb5_magic magic;

    krb5_msgtype msg_type;
    krb5_pa_data **padata;
    krb5_principal client;
    krb5_ticket *ticket;
    krb5_enc_data enc_part;

    krb5_enc_kdc_rep_part *enc_part2;
} krb5_kdc_rep;


typedef struct _krb5_error {
    krb5_magic magic;

    krb5_timestamp ctime;
    krb5_int32 cusec;
    krb5_int32 susec;
    krb5_timestamp stime;
    krb5_ui_4 error;
    krb5_principal client;

    krb5_principal server;
    krb5_data text;
    krb5_data e_data;
} krb5_error;

typedef struct _krb5_ap_req {
    krb5_magic magic;
    krb5_flags ap_options;
    krb5_ticket *ticket;
    krb5_enc_data authenticator;
} krb5_ap_req;

typedef struct _krb5_ap_rep {
    krb5_magic magic;
    krb5_enc_data enc_part;
} krb5_ap_rep;

typedef struct _krb5_ap_rep_enc_part {
    krb5_magic magic;
    krb5_timestamp ctime;
    krb5_int32 cusec;
    krb5_keyblock *subkey;
    krb5_ui_4 seq_number;
} krb5_ap_rep_enc_part;

typedef struct _krb5_response {
    krb5_magic magic;
    krb5_octet message_type;
    krb5_data response;
    krb5_int32 expected_nonce;
    krb5_timestamp request_time;
} krb5_response;

typedef struct _krb5_cred_info {
    krb5_magic magic;
    krb5_keyblock *session;

    krb5_principal client;
    krb5_principal server;
    krb5_flags flags;
    krb5_ticket_times times;

    krb5_address **caddrs;
} krb5_cred_info;

typedef struct _krb5_cred_enc_part {
    krb5_magic magic;
    krb5_int32 nonce;
    krb5_timestamp timestamp;
    krb5_int32 usec;
    krb5_address *s_address;
    krb5_address *r_address;
    krb5_cred_info **ticket_info;
} krb5_cred_enc_part;

typedef struct _krb5_cred {
    krb5_magic magic;
    krb5_ticket **tickets;
    krb5_enc_data enc_part;
    krb5_cred_enc_part *enc_part2;
} krb5_cred;


typedef struct _passwd_phrase_element {
    krb5_magic magic;
    krb5_data *passwd;
    krb5_data *phrase;
} passwd_phrase_element;

typedef struct _krb5_pwd_data {
    krb5_magic magic;
    int sequence_count;
    passwd_phrase_element **element;
} krb5_pwd_data;






typedef struct _krb5_pa_svr_referral_data {

    krb5_principal principal;
} krb5_pa_svr_referral_data;

typedef struct _krb5_pa_server_referral_data {
    krb5_data *referred_realm;
    krb5_principal true_principal_name;
    krb5_principal requested_principal_name;
    krb5_timestamp referral_valid_until;
    krb5_checksum rep_cksum;
} krb5_pa_server_referral_data;

typedef struct _krb5_pa_pac_req {

    krb5_boolean include_pac;
} krb5_pa_pac_req;
# 1386 "/usr/include/krb5/krb5.h" 3 4
typedef struct krb5_replay_data {
    krb5_timestamp timestamp;
    krb5_int32 usec;
    krb5_ui_4 seq;
} krb5_replay_data;
# 1401 "/usr/include/krb5/krb5.h" 3 4
typedef krb5_error_code
( * krb5_mk_req_checksum_func) (krb5_context, krb5_auth_context , void *,
                                             krb5_data **);
# 1414 "/usr/include/krb5/krb5.h" 3 4
typedef krb5_pointer krb5_cc_cursor;

struct _krb5_ccache;
typedef struct _krb5_ccache *krb5_ccache;
struct _krb5_cc_ops;
typedef struct _krb5_cc_ops krb5_cc_ops;




struct _krb5_cccol_cursor;
typedef struct _krb5_cccol_cursor *krb5_cccol_cursor;
# 1443 "/usr/include/krb5/krb5.h" 3 4
const char *
krb5_cc_get_name(krb5_context context, krb5_ccache cache);

krb5_error_code
krb5_cc_gen_new (krb5_context context, krb5_ccache *cache);

krb5_error_code
krb5_cc_initialize(krb5_context context, krb5_ccache cache,
                   krb5_principal principal);

krb5_error_code
krb5_cc_destroy(krb5_context context, krb5_ccache cache);

krb5_error_code
krb5_cc_close(krb5_context context, krb5_ccache cache);

krb5_error_code
krb5_cc_store_cred(krb5_context context, krb5_ccache cache, krb5_creds *creds);

krb5_error_code
krb5_cc_retrieve_cred(krb5_context context, krb5_ccache cache,
                      krb5_flags flags, krb5_creds *mcreds,
                      krb5_creds *creds);

krb5_error_code
krb5_cc_get_principal(krb5_context context, krb5_ccache cache,
                      krb5_principal *principal);

krb5_error_code
krb5_cc_start_seq_get(krb5_context context, krb5_ccache cache,
                      krb5_cc_cursor *cursor);

krb5_error_code
krb5_cc_next_cred(krb5_context context, krb5_ccache cache,
                  krb5_cc_cursor *cursor, krb5_creds *creds);

krb5_error_code
krb5_cc_end_seq_get(krb5_context context, krb5_ccache cache,
                    krb5_cc_cursor *cursor);

krb5_error_code
krb5_cc_remove_cred(krb5_context context, krb5_ccache cache, krb5_flags flags,
                    krb5_creds *creds);

krb5_error_code
krb5_cc_set_flags(krb5_context context, krb5_ccache cache, krb5_flags flags);

krb5_error_code
krb5_cc_get_flags(krb5_context context, krb5_ccache cache, krb5_flags *flags);

const char *
krb5_cc_get_type(krb5_context context, krb5_ccache cache);

krb5_error_code
krb5_cc_move(krb5_context context, krb5_ccache src, krb5_ccache dst);

krb5_error_code
krb5_cc_last_change_time(krb5_context context, krb5_ccache ccache,
                         krb5_timestamp *change_time);

krb5_error_code
krb5_cc_lock(krb5_context context, krb5_ccache ccache);

krb5_error_code
krb5_cc_unlock(krb5_context context, krb5_ccache ccache);

krb5_error_code
krb5_cccol_cursor_new(krb5_context context, krb5_cccol_cursor *cursor);

krb5_error_code
krb5_cccol_cursor_next(krb5_context context, krb5_cccol_cursor cursor,
                       krb5_ccache *ccache);

krb5_error_code
krb5_cccol_cursor_free(krb5_context context, krb5_cccol_cursor *cursor);

krb5_error_code
krb5_cccol_last_change_time(krb5_context context, krb5_timestamp *change_time);

krb5_error_code
krb5_cccol_lock(krb5_context context);

krb5_error_code
krb5_cccol_unlock(krb5_context context);

krb5_error_code
krb5_cc_new_unique(krb5_context context, const char *type, const char *hint,
                   krb5_ccache *id);
# 1540 "/usr/include/krb5/krb5.h" 3 4
struct krb5_rc_st;
typedef struct krb5_rc_st *krb5_rcache;
# 1555 "/usr/include/krb5/krb5.h" 3 4
typedef krb5_pointer krb5_kt_cursor;

typedef struct krb5_keytab_entry_st {
    krb5_magic magic;
    krb5_principal principal;
    krb5_timestamp timestamp;
    krb5_kvno vno;
    krb5_keyblock key;
} krb5_keytab_entry;

struct _krb5_kt;
typedef struct _krb5_kt *krb5_keytab;

const char *
krb5_kt_get_type(krb5_context, krb5_keytab keytab);

krb5_error_code
krb5_kt_get_name(krb5_context context, krb5_keytab keytab, char *name,
                 unsigned int namelen);

krb5_error_code
krb5_kt_close(krb5_context context, krb5_keytab keytab);

krb5_error_code
krb5_kt_get_entry(krb5_context context, krb5_keytab keytab,
                  krb5_const_principal principal, krb5_kvno vno,
                  krb5_enctype enctype, krb5_keytab_entry *entry);

krb5_error_code
krb5_kt_start_seq_get(krb5_context context, krb5_keytab keytab,
                      krb5_kt_cursor *cursor);

krb5_error_code
krb5_kt_next_entry(krb5_context context, krb5_keytab keytab,
                   krb5_keytab_entry *entry, krb5_kt_cursor *cursor);

krb5_error_code
krb5_kt_end_seq_get(krb5_context context, krb5_keytab keytab,
                    krb5_kt_cursor *cursor);
# 1603 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code krb5_init_context(krb5_context *);
krb5_error_code krb5_init_secure_context(krb5_context *);
void krb5_free_context(krb5_context);
krb5_error_code krb5_copy_context(krb5_context, krb5_context *);

krb5_error_code
krb5_set_default_tgs_enctypes(krb5_context, const krb5_enctype *);

krb5_error_code
krb5_get_permitted_enctypes(krb5_context, krb5_enctype **);

krb5_boolean krb5_is_thread_safe(void);



krb5_error_code
krb5_server_decrypt_ticket_keytab(krb5_context context, const krb5_keytab kt,
                                  krb5_ticket *ticket);

void krb5_free_tgt_creds(krb5_context, krb5_creds **);
# 1632 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_get_credentials(krb5_context, krb5_flags, krb5_ccache, krb5_creds *,
                     krb5_creds **);

krb5_error_code
krb5_get_credentials_validate(krb5_context, krb5_flags, krb5_ccache,
                              krb5_creds *, krb5_creds **);

krb5_error_code
krb5_get_credentials_renew(krb5_context, krb5_flags, krb5_ccache, krb5_creds *,
                           krb5_creds **);

krb5_error_code
krb5_mk_req(krb5_context, krb5_auth_context *, krb5_flags, char *, char *,
            krb5_data *, krb5_ccache, krb5_data *);

krb5_error_code
krb5_mk_req_extended(krb5_context, krb5_auth_context *, krb5_flags,
                     krb5_data *, krb5_creds *, krb5_data * );

krb5_error_code
krb5_mk_rep(krb5_context, krb5_auth_context, krb5_data *);

krb5_error_code
krb5_mk_rep_dce(krb5_context, krb5_auth_context, krb5_data *);

krb5_error_code
krb5_rd_rep(krb5_context, krb5_auth_context, const krb5_data *,
            krb5_ap_rep_enc_part **);

krb5_error_code
krb5_rd_rep_dce(krb5_context, krb5_auth_context, const krb5_data *,
                krb5_ui_4 *);

krb5_error_code
krb5_mk_error(krb5_context, const krb5_error *, krb5_data *);

krb5_error_code
krb5_rd_error(krb5_context, const krb5_data *, krb5_error **);

krb5_error_code
krb5_rd_safe(krb5_context, krb5_auth_context, const krb5_data *, krb5_data *,
             krb5_replay_data *);

krb5_error_code
krb5_rd_priv(krb5_context, krb5_auth_context, const krb5_data *, krb5_data *,
             krb5_replay_data *);

krb5_error_code
krb5_parse_name(krb5_context, const char *, krb5_principal *);




krb5_error_code
krb5_parse_name_flags(krb5_context, const char *, int, krb5_principal *);

krb5_error_code
krb5_unparse_name(krb5_context, krb5_const_principal, char **);

krb5_error_code
krb5_unparse_name_ext(krb5_context, krb5_const_principal, char **,
                      unsigned int *);




krb5_error_code
krb5_unparse_name_flags(krb5_context, krb5_const_principal, int, char **);

krb5_error_code
krb5_unparse_name_flags_ext(krb5_context, krb5_const_principal, int,
                            char **, unsigned int *);

krb5_error_code
krb5_set_principal_realm(krb5_context, krb5_principal, const char *);

krb5_boolean
krb5_address_search(krb5_context, const krb5_address *, krb5_address *const *);

krb5_boolean
krb5_address_compare(krb5_context, const krb5_address *, const krb5_address *);

int
krb5_address_order(krb5_context, const krb5_address *, const krb5_address *);

krb5_boolean
krb5_realm_compare(krb5_context, krb5_const_principal, krb5_const_principal);

krb5_boolean
krb5_principal_compare(krb5_context, krb5_const_principal,
                       krb5_const_principal);

krb5_boolean
krb5_principal_compare_any_realm(krb5_context, krb5_const_principal,
                                 krb5_const_principal);






krb5_boolean
krb5_principal_compare_flags(krb5_context, krb5_const_principal,
                             krb5_const_principal, int);

krb5_error_code
krb5_init_keyblock(krb5_context, krb5_enctype enctype, size_t length,
                   krb5_keyblock **out);







krb5_error_code
krb5_copy_keyblock(krb5_context, const krb5_keyblock *, krb5_keyblock **);

krb5_error_code
krb5_copy_keyblock_contents(krb5_context, const krb5_keyblock *,
                            krb5_keyblock *);

krb5_error_code
krb5_copy_creds(krb5_context, const krb5_creds *, krb5_creds **);

krb5_error_code
krb5_copy_data(krb5_context, const krb5_data *, krb5_data **);

krb5_error_code
krb5_copy_principal(krb5_context, krb5_const_principal, krb5_principal *);

krb5_error_code
krb5_copy_addresses(krb5_context, krb5_address * const *, krb5_address ***);

krb5_error_code
krb5_copy_ticket(krb5_context, const krb5_ticket *, krb5_ticket **);

krb5_error_code
krb5_copy_authdata(krb5_context, krb5_authdata * const *, krb5_authdata ***);



krb5_error_code
krb5_merge_authdata(krb5_context, krb5_authdata * const *,
                    krb5_authdata *const *, krb5_authdata ***);

krb5_error_code
krb5_copy_authenticator(krb5_context, const krb5_authenticator *,
                        krb5_authenticator **);

krb5_error_code
krb5_copy_checksum(krb5_context, const krb5_checksum *, krb5_checksum **);

krb5_error_code
krb5_get_server_rcache(krb5_context, const krb5_data *, krb5_rcache *);

krb5_error_code
krb5_build_principal_ext(krb5_context, krb5_principal *, unsigned int,
                         const char *, ...);

krb5_error_code
krb5_build_principal(krb5_context, krb5_principal *, unsigned int,
                     const char *, ...)

    __attribute__ ((sentinel))

    ;







krb5_error_code
krb5_build_principal_alloc_va(krb5_context, krb5_principal *, unsigned int,
                              const char *, va_list);

krb5_error_code
krb5_425_conv_principal(krb5_context, const char *name, const char *instance,
                        const char *realm, krb5_principal *princ);

krb5_error_code
krb5_524_conv_principal(krb5_context context, krb5_const_principal princ,
                        char *name, char *inst, char *realm);

struct credentials;
int
krb5_524_convert_creds(krb5_context context, krb5_creds *v5creds,
                       struct credentials *v4creds);






krb5_error_code
krb5_kt_resolve(krb5_context, const char *, krb5_keytab *);

krb5_error_code
krb5_kt_default_name(krb5_context, char *, int);

krb5_error_code
krb5_kt_default(krb5_context, krb5_keytab * );

krb5_error_code
krb5_free_keytab_entry_contents(krb5_context, krb5_keytab_entry *);



krb5_error_code krb5_kt_free_entry(krb5_context,
                                                 krb5_keytab_entry * );



krb5_error_code
krb5_kt_remove_entry(krb5_context, krb5_keytab, krb5_keytab_entry *);

krb5_error_code
krb5_kt_add_entry(krb5_context, krb5_keytab, krb5_keytab_entry *);

krb5_error_code
krb5_principal2salt(krb5_context, krb5_const_principal, krb5_data *);



krb5_error_code
krb5_cc_resolve(krb5_context, const char *, krb5_ccache *);

const char *
krb5_cc_default_name(krb5_context);

krb5_error_code
krb5_cc_set_default_name(krb5_context, const char *);

krb5_error_code
krb5_cc_default(krb5_context, krb5_ccache *);

krb5_error_code
krb5_cc_copy_creds(krb5_context context, krb5_ccache incc, krb5_ccache outcc);

krb5_error_code
krb5_cc_get_config(krb5_context, krb5_ccache,
                   krb5_const_principal,
                   const char *, krb5_data *);

krb5_error_code
krb5_cc_set_config(krb5_context, krb5_ccache,
                   krb5_const_principal,
                   const char *, krb5_data *);

krb5_boolean
krb5_is_config_principal(krb5_context,
                         krb5_const_principal);


void krb5_free_principal(krb5_context, krb5_principal );
void krb5_free_authenticator(krb5_context,
                                           krb5_authenticator * );
void krb5_free_addresses(krb5_context, krb5_address ** );
void krb5_free_authdata(krb5_context, krb5_authdata ** );
void krb5_free_ticket(krb5_context, krb5_ticket * );
void krb5_free_error(krb5_context, krb5_error * );
void krb5_free_creds(krb5_context, krb5_creds *);
void krb5_free_cred_contents(krb5_context, krb5_creds *);
void krb5_free_checksum(krb5_context, krb5_checksum *);
void krb5_free_checksum_contents(krb5_context, krb5_checksum *);
void krb5_free_keyblock(krb5_context, krb5_keyblock *);
void krb5_free_keyblock_contents(krb5_context, krb5_keyblock *);
void krb5_free_ap_rep_enc_part(krb5_context,
                                             krb5_ap_rep_enc_part *);
void krb5_free_data(krb5_context, krb5_data *);
void krb5_free_data_contents(krb5_context, krb5_data *);
void krb5_free_unparsed_name(krb5_context, char *);
void krb5_free_cksumtypes(krb5_context, krb5_cksumtype *);


krb5_error_code
krb5_us_timeofday(krb5_context, krb5_timestamp *, krb5_int32 *);

krb5_error_code
krb5_timeofday(krb5_context, krb5_timestamp *);


krb5_error_code
krb5_os_localaddr(krb5_context, krb5_address ***);

krb5_error_code
krb5_get_default_realm(krb5_context, char **);

krb5_error_code
krb5_set_default_realm(krb5_context, const char * );

void
krb5_free_default_realm(krb5_context, char * );

krb5_error_code
krb5_sname_to_principal(krb5_context, const char *, const char *, krb5_int32,
                        krb5_principal *);

krb5_error_code
krb5_change_password(krb5_context context, krb5_creds *creds, char *newpw,
                     int *result_code, krb5_data *result_code_string,
                     krb5_data *result_string);

krb5_error_code
krb5_set_password(krb5_context context, krb5_creds *creds, char *newpw,
                  krb5_principal change_password_for, int *result_code,
                  krb5_data *result_code_string, krb5_data *result_string);

krb5_error_code
krb5_set_password_using_ccache(krb5_context context, krb5_ccache ccache,
                               char *newpw, krb5_principal change_password_for,
                               int *result_code, krb5_data *result_code_string,
                               krb5_data *result_string);

krb5_error_code
krb5_get_profile(krb5_context, struct _profile_t * *);
# 1981 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_rd_req(krb5_context, krb5_auth_context *, const krb5_data *,
            krb5_const_principal, krb5_keytab, krb5_flags *, krb5_ticket **);

krb5_error_code
krb5_kt_read_service_key(krb5_context, krb5_pointer, krb5_principal, krb5_kvno,
                         krb5_enctype, krb5_keyblock **);

krb5_error_code
krb5_mk_safe(krb5_context, krb5_auth_context, const krb5_data *, krb5_data *,
             krb5_replay_data *);

krb5_error_code
krb5_mk_priv(krb5_context, krb5_auth_context, const krb5_data *, krb5_data *,
             krb5_replay_data *);

krb5_error_code
krb5_sendauth(krb5_context, krb5_auth_context *, krb5_pointer, char *,
              krb5_principal, krb5_principal, krb5_flags, krb5_data *,
              krb5_creds *, krb5_ccache, krb5_error **,
              krb5_ap_rep_enc_part **, krb5_creds **);

krb5_error_code
krb5_recvauth(krb5_context, krb5_auth_context *, krb5_pointer, char *,
              krb5_principal, krb5_int32, krb5_keytab, krb5_ticket **);

krb5_error_code
krb5_recvauth_version(krb5_context, krb5_auth_context *, krb5_pointer,
                      krb5_principal, krb5_int32, krb5_keytab, krb5_ticket **,
                      krb5_data *);

krb5_error_code
krb5_mk_ncred(krb5_context, krb5_auth_context, krb5_creds **, krb5_data **,
              krb5_replay_data *);

krb5_error_code
krb5_mk_1cred(krb5_context, krb5_auth_context, krb5_creds *, krb5_data **,
              krb5_replay_data *);

krb5_error_code
krb5_rd_cred(krb5_context, krb5_auth_context, krb5_data *, krb5_creds ***,
             krb5_replay_data *);

krb5_error_code
krb5_fwd_tgt_creds(krb5_context, krb5_auth_context, char *, krb5_principal,
                   krb5_principal, krb5_ccache, int forwardable, krb5_data *);

krb5_error_code
krb5_auth_con_init(krb5_context, krb5_auth_context *);

krb5_error_code
krb5_auth_con_free(krb5_context, krb5_auth_context);

krb5_error_code
krb5_auth_con_setflags(krb5_context, krb5_auth_context, krb5_int32);

krb5_error_code
krb5_auth_con_getflags(krb5_context, krb5_auth_context, krb5_int32 *);

krb5_error_code
krb5_auth_con_set_checksum_func(krb5_context, krb5_auth_context,
                                krb5_mk_req_checksum_func, void *);

krb5_error_code
krb5_auth_con_get_checksum_func(krb5_context, krb5_auth_context,
                                krb5_mk_req_checksum_func *, void **);

krb5_error_code
krb5_auth_con_setaddrs(krb5_context, krb5_auth_context, krb5_address *,
                       krb5_address *);

krb5_error_code
krb5_auth_con_getaddrs(krb5_context, krb5_auth_context, krb5_address **,
                       krb5_address **);

krb5_error_code
krb5_auth_con_setports(krb5_context, krb5_auth_context, krb5_address *,
                       krb5_address *);

krb5_error_code
krb5_auth_con_setuseruserkey(krb5_context, krb5_auth_context, krb5_keyblock *);

krb5_error_code
krb5_auth_con_getkey(krb5_context, krb5_auth_context, krb5_keyblock **);

krb5_error_code
krb5_auth_con_getkey_k(krb5_context, krb5_auth_context, krb5_key *);

krb5_error_code
krb5_auth_con_getsendsubkey(krb5_context, krb5_auth_context, krb5_keyblock **);

krb5_error_code
krb5_auth_con_getsendsubkey_k(krb5_context, krb5_auth_context, krb5_key *);

krb5_error_code
krb5_auth_con_getrecvsubkey(krb5_context, krb5_auth_context, krb5_keyblock **);

krb5_error_code
krb5_auth_con_getrecvsubkey_k(krb5_context, krb5_auth_context, krb5_key *);

krb5_error_code
krb5_auth_con_setsendsubkey(krb5_context, krb5_auth_context, krb5_keyblock *);

krb5_error_code
krb5_auth_con_setrecvsubkey(krb5_context, krb5_auth_context, krb5_keyblock *);
# 2097 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_auth_con_getlocalseqnumber(krb5_context, krb5_auth_context, krb5_int32 *);

krb5_error_code
krb5_auth_con_getremoteseqnumber(krb5_context, krb5_auth_context,
                                 krb5_int32 *);






krb5_error_code
krb5_auth_con_setrcache(krb5_context, krb5_auth_context, krb5_rcache);

krb5_error_code
krb5_auth_con_getrcache(krb5_context, krb5_auth_context, krb5_rcache *);

krb5_error_code
krb5_auth_con_getauthenticator(krb5_context, krb5_auth_context,
                               krb5_authenticator **);

krb5_error_code
krb5_auth_con_set_req_cksumtype(krb5_context, krb5_auth_context,
                                krb5_cksumtype);
# 2133 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_read_password(krb5_context, const char *, const char *, char *,
                   unsigned int * );

krb5_error_code
krb5_aname_to_localname(krb5_context, krb5_const_principal, int, char *);

krb5_error_code
krb5_get_host_realm(krb5_context, const char *, char ***);

krb5_error_code
krb5_get_fallback_host_realm(krb5_context, krb5_data *, char ***);

krb5_error_code
krb5_free_host_realm(krb5_context, char * const * );

krb5_boolean
krb5_kuserok(krb5_context, krb5_principal, const char *);

krb5_error_code
krb5_auth_con_genaddrs(krb5_context, krb5_auth_context, int, int);

krb5_error_code
krb5_set_real_time(krb5_context, krb5_timestamp, krb5_int32);

krb5_error_code
krb5_get_time_offsets(krb5_context, krb5_timestamp *, krb5_int32 *);


krb5_error_code krb5_string_to_enctype(char *, krb5_enctype *);
krb5_error_code krb5_string_to_salttype(char *, krb5_int32 *);
krb5_error_code krb5_string_to_cksumtype(char *,
                                                       krb5_cksumtype *);
krb5_error_code krb5_string_to_timestamp(char *,
                                                       krb5_timestamp *);
krb5_error_code krb5_string_to_deltat(char *, krb5_deltat *);
krb5_error_code krb5_enctype_to_string(krb5_enctype, char *,
                                                     size_t);
krb5_error_code krb5_salttype_to_string(krb5_int32, char *,
                                                      size_t);
krb5_error_code krb5_cksumtype_to_string(krb5_cksumtype, char *,
                                                       size_t);
krb5_error_code krb5_timestamp_to_string(krb5_timestamp, char *,
                                                       size_t);
krb5_error_code krb5_timestamp_to_sfstring(krb5_timestamp,
                                                         char *, size_t,
                                                         char *);
krb5_error_code krb5_deltat_to_string(krb5_deltat, char *,
                                                    size_t);
# 2194 "/usr/include/krb5/krb5.h" 3 4
typedef struct _krb5_prompt {
    char *prompt;
    int hidden;
    krb5_data *reply;
} krb5_prompt;

typedef krb5_error_code
( *krb5_prompter_fct)(krb5_context context, void *data,
                                   const char *name, const char *banner,
                                   int num_prompts, krb5_prompt prompts[]);

krb5_error_code
krb5_prompter_posix(krb5_context context, void *data, const char *name,
                    const char *banner, int num_prompts,
                    krb5_prompt prompts[]);

typedef struct _krb5_get_init_creds_opt {
    krb5_flags flags;
    krb5_deltat tkt_life;
    krb5_deltat renew_life;
    int forwardable;
    int proxiable;
    krb5_enctype *etype_list;
    int etype_list_length;
    krb5_address **address_list;
    krb5_preauthtype *preauth_list;
    int preauth_list_length;
    krb5_data *salt;
} krb5_get_init_creds_opt;
# 2237 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_get_init_creds_opt_alloc(krb5_context context,
                              krb5_get_init_creds_opt **opt);

void
krb5_get_init_creds_opt_free(krb5_context context,
                             krb5_get_init_creds_opt *opt);

void
krb5_get_init_creds_opt_init(krb5_get_init_creds_opt *opt);

void
krb5_get_init_creds_opt_set_tkt_life(krb5_get_init_creds_opt *opt,
                                     krb5_deltat tkt_life);

void
krb5_get_init_creds_opt_set_renew_life(krb5_get_init_creds_opt *opt,
                                       krb5_deltat renew_life);

void
krb5_get_init_creds_opt_set_forwardable(krb5_get_init_creds_opt *opt,
                                        int forwardable);

void
krb5_get_init_creds_opt_set_proxiable(krb5_get_init_creds_opt *opt,
                                      int proxiable);

void
krb5_get_init_creds_opt_set_canonicalize(krb5_get_init_creds_opt *opt,
                                         int canonicalize);
# 2279 "/usr/include/krb5/krb5.h" 3 4
void
krb5_get_init_creds_opt_set_anonymous(krb5_get_init_creds_opt *opt,
                                      int anonymous);

void
krb5_get_init_creds_opt_set_etype_list(krb5_get_init_creds_opt *opt,
                                       krb5_enctype *etype_list,
                                       int etype_list_length);

void
krb5_get_init_creds_opt_set_address_list(krb5_get_init_creds_opt *opt,
                                         krb5_address **addresses);

void
krb5_get_init_creds_opt_set_preauth_list(krb5_get_init_creds_opt *opt,
                                         krb5_preauthtype *preauth_list,
                                         int preauth_list_length);

void
krb5_get_init_creds_opt_set_salt(krb5_get_init_creds_opt *opt,
                                 krb5_data *salt);

void
krb5_get_init_creds_opt_set_change_password_prompt(krb5_get_init_creds_opt
                                                   *opt, int prompt);


typedef struct _krb5_gic_opt_pa_data {
    char *attr;
    char *value;
} krb5_gic_opt_pa_data;
# 2319 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_get_init_creds_opt_set_pa(krb5_context context,
                               krb5_get_init_creds_opt *opt, const char *attr,
                               const char *value);
# 2335 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_get_init_creds_opt_set_fast_ccache_name(krb5_context context,
                                             krb5_get_init_creds_opt *opt,
                                             const char *fast_ccache_name);
# 2347 "/usr/include/krb5/krb5.h" 3 4
krb5_error_code
krb5_get_init_creds_opt_set_out_ccache(krb5_context context,
                                       krb5_get_init_creds_opt *opt,
                                       krb5_ccache ccache);

krb5_error_code
krb5_get_init_creds_opt_set_fast_flags(krb5_context context,
                                       krb5_get_init_creds_opt *opt,
                                       krb5_flags flags);

krb5_error_code
krb5_get_init_creds_opt_get_fast_flags(krb5_context context,
                                       krb5_get_init_creds_opt *opt,
                                       krb5_flags *out_flags);




krb5_error_code
krb5_get_init_creds_password(krb5_context context, krb5_creds *creds,
                             krb5_principal client, char *password,
                             krb5_prompter_fct prompter, void *data,
                             krb5_deltat start_time, char *in_tkt_service,
                             krb5_get_init_creds_opt *k5_gic_options);

struct _krb5_init_creds_context;
typedef struct _krb5_init_creds_context *krb5_init_creds_context;

void
krb5_init_creds_free(krb5_context context, krb5_init_creds_context ctx);

krb5_error_code
krb5_init_creds_get(krb5_context context, krb5_init_creds_context ctx);

krb5_error_code
krb5_init_creds_get_creds(krb5_context context, krb5_init_creds_context ctx,
                          krb5_creds *creds);

krb5_error_code
krb5_init_creds_get_error(krb5_context context, krb5_init_creds_context ctx,
                          krb5_error **error);

krb5_error_code
krb5_init_creds_init(krb5_context context, krb5_principal client,
                     krb5_prompter_fct prompter, void *data,
                     krb5_deltat start_time, krb5_get_init_creds_opt *options,
                     krb5_init_creds_context *ctx);

krb5_error_code
krb5_init_creds_set_keyblock(krb5_context context, krb5_init_creds_context ctx,
                             krb5_keyblock *keyblock);

krb5_error_code
krb5_init_creds_set_keytab(krb5_context context, krb5_init_creds_context ctx,
                           krb5_keytab keytab);

krb5_error_code
krb5_init_creds_step(krb5_context context, krb5_init_creds_context ctx,
                     krb5_data *in, krb5_data *out, krb5_data *realm,
                     unsigned int *flags);

krb5_error_code
krb5_init_creds_set_password(krb5_context context, krb5_init_creds_context ctx,
                             const char *password);

krb5_error_code
krb5_init_creds_set_service(krb5_context context, krb5_init_creds_context ctx,
                            const char *service);

krb5_error_code
krb5_init_creds_get_times(krb5_context context, krb5_init_creds_context ctx,
                          krb5_ticket_times *times);

krb5_error_code
krb5_get_init_creds_keytab(krb5_context context, krb5_creds *creds,
                           krb5_principal client, krb5_keytab arg_keytab,
                           krb5_deltat start_time, char *in_tkt_service,
                           krb5_get_init_creds_opt *k5_gic_options);

typedef struct _krb5_verify_init_creds_opt {
    krb5_flags flags;
    int ap_req_nofail;
} krb5_verify_init_creds_opt;



void
krb5_verify_init_creds_opt_init(krb5_verify_init_creds_opt *k5_vic_options);

void
krb5_verify_init_creds_opt_set_ap_req_nofail(krb5_verify_init_creds_opt *
                                             k5_vic_options,
                                             int ap_req_nofail);

krb5_error_code
krb5_verify_init_creds(krb5_context context, krb5_creds *creds,
                       krb5_principal ap_req_server, krb5_keytab ap_req_keytab,
                       krb5_ccache *ccache,
                       krb5_verify_init_creds_opt *k5_vic_options);

krb5_error_code
krb5_get_validated_creds(krb5_context context, krb5_creds *creds,
                         krb5_principal client, krb5_ccache ccache,
                         char *in_tkt_service);

krb5_error_code
krb5_get_renewed_creds(krb5_context context, krb5_creds *creds,
                       krb5_principal client, krb5_ccache ccache,
                       char *in_tkt_service);

krb5_error_code
krb5_decode_ticket(const krb5_data *code, krb5_ticket **rep);

void
krb5_appdefault_string(krb5_context context, const char *appname,
                       const krb5_data *realm, const char *option,
                       const char *default_value, char ** ret_value);

void
krb5_appdefault_boolean(krb5_context context, const char *appname,
                        const krb5_data *realm, const char *option,
                        int default_value, int *ret_value);
# 2479 "/usr/include/krb5/krb5.h" 3 4
typedef krb5_int32 krb5_prompt_type;

krb5_prompt_type* krb5_get_prompt_types(krb5_context context);


void
krb5_set_error_message(krb5_context, krb5_error_code, const char *, ...)

    __attribute__((__format__(__printf__, 3, 4)))

    ;
void
krb5_vset_error_message(krb5_context, krb5_error_code, const char *, va_list)

    __attribute__((__format__(__printf__, 3, 0)))

    ;
void
krb5_copy_error_message(krb5_context, krb5_context);
# 2509 "/usr/include/krb5/krb5.h" 3 4
const char *
krb5_get_error_message(krb5_context, krb5_error_code);
void
krb5_free_error_message(krb5_context, const char *);
void
krb5_clear_error_message(krb5_context);

krb5_error_code
krb5_decode_authdata_container(krb5_context context,
                               krb5_authdatatype type,
                               const krb5_authdata *container,
                               krb5_authdata ***authdata);

krb5_error_code
krb5_encode_authdata_container(krb5_context context,
                               krb5_authdatatype type,
                               krb5_authdata * const*authdata,
                               krb5_authdata ***container);




krb5_error_code
krb5_make_authdata_kdc_issued(krb5_context context,
                              const krb5_keyblock *key,
                              krb5_const_principal issuer,
                              krb5_authdata *const *authdata,
                              krb5_authdata ***ad_kdcissued);

krb5_error_code
krb5_verify_authdata_kdc_issued(krb5_context context,
                                const krb5_keyblock *key,
                                const krb5_authdata *ad_kdcissued,
                                krb5_principal *issuer,
                                krb5_authdata ***authdata);




struct krb5_pac_data;
typedef struct krb5_pac_data *krb5_pac;

krb5_error_code
krb5_pac_add_buffer(krb5_context context, krb5_pac pac, krb5_ui_4 type,
                    const krb5_data *data);

void
krb5_pac_free(krb5_context context, krb5_pac pac);

krb5_error_code
krb5_pac_get_buffer(krb5_context context, krb5_pac pac, krb5_ui_4 type,
                    krb5_data *data);

krb5_error_code
krb5_pac_get_types(krb5_context context, krb5_pac pac, size_t *len,
                   krb5_ui_4 **types);

krb5_error_code
krb5_pac_init(krb5_context context, krb5_pac *pac);

krb5_error_code
krb5_pac_parse(krb5_context context, const void *ptr, size_t len,
               krb5_pac *pac);

krb5_error_code
krb5_pac_verify(krb5_context context, const krb5_pac pac,
                krb5_timestamp authtime, krb5_const_principal principal,
                const krb5_keyblock *server, const krb5_keyblock *privsvr);



krb5_error_code
krb5_allow_weak_crypto(krb5_context context, krb5_boolean enable);






# 2601 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/include/et/com_err.h" 1 3 4
# 19 "/usr/include/et/com_err.h" 3 4
# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 1 3 4
# 149 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stddef.h" 3 4
typedef long int ptrdiff_t;
# 20 "/usr/include/et/com_err.h" 2 3 4


typedef long errcode_t;

struct error_table {
 char const * const * msgs;
 long base;
 int n_msgs;
};
struct et_list;

extern void com_err (const char *, long, const char *, ...)
 __attribute__((format(printf, 3, 4)));

extern void com_err_va (const char *whoami, errcode_t code, const char *fmt,
   va_list args)
 __attribute__((format(printf, 3, 0)));

extern char const *error_message (long);
extern void (*com_err_hook) (const char *, long, const char *, va_list);
extern void (*set_com_err_hook (void (*) (const char *, long,
       const char *, va_list)))
 (const char *, long, const char *, va_list);
extern void (*reset_com_err_hook (void)) (const char *, long,
       const char *, va_list);
extern int init_error_table(const char * const *msgs, long base, int count);

extern errcode_t add_error_table(const struct error_table * et);
extern errcode_t remove_error_table(const struct error_table * et);
extern void add_to_error_table(struct et_list *new_table);


extern const char *com_right(struct et_list *list, long code);
extern const char *com_right_r(struct et_list *list, long code, char *str, size_t len);
extern void initialize_error_table_r(struct et_list **list,
         const char **messages,
         int num_errors,
         long base);
extern void free_error_table(struct et_list *et);


extern int et_list_lock(void);
extern int et_list_unlock(void);
# 2602 "/usr/include/krb5/krb5.h" 2 3 4
# 2858 "/usr/include/krb5/krb5.h" 3 4
extern const struct error_table et_krb5_error_table;
extern void initialize_krb5_error_table(void);


extern void initialize_krb5_error_table_r(struct et_list **list);
# 2874 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/include/et/com_err.h" 1 3 4
# 2875 "/usr/include/krb5/krb5.h" 2 3 4
# 2919 "/usr/include/krb5/krb5.h" 3 4
extern const struct error_table et_kdb5_error_table;
extern void initialize_kdb5_error_table(void);


extern void initialize_kdb5_error_table_r(struct et_list **list);
# 2935 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/include/et/com_err.h" 1 3 4
# 2936 "/usr/include/krb5/krb5.h" 2 3 4
# 2998 "/usr/include/krb5/krb5.h" 3 4
extern const struct error_table et_kv5m_error_table;
extern void initialize_kv5m_error_table(void);


extern void initialize_kv5m_error_table_r(struct et_list **list);
# 3014 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/include/et/com_err.h" 1 3 4
# 3015 "/usr/include/krb5/krb5.h" 2 3 4
# 3025 "/usr/include/krb5/krb5.h" 3 4
extern const struct error_table et_k524_error_table;
extern void initialize_k524_error_table(void);


extern void initialize_k524_error_table_r(struct et_list **list);
# 3041 "/usr/include/krb5/krb5.h" 3 4
# 1 "/usr/include/et/com_err.h" 1 3 4
# 3042 "/usr/include/krb5/krb5.h" 2 3 4
# 3056 "/usr/include/krb5/krb5.h" 3 4
extern const struct error_table et_asn1_error_table;
extern void initialize_asn1_error_table(void);


extern void initialize_asn1_error_table_r(struct et_list **list);
# 12 "git-remote-gitkrb5.c" 2

# 1 "gitkrb5-client.h" 1


# 1 "../common/common.h" 1

# 1 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdint.h" 1 3 4


# 1 "/usr/include/stdint.h" 1 3 4
# 27 "/usr/include/stdint.h" 3 4
# 1 "/usr/include/bits/wchar.h" 1 3 4
# 28 "/usr/include/stdint.h" 2 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 29 "/usr/include/stdint.h" 2 3 4
# 49 "/usr/include/stdint.h" 3 4
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;

typedef unsigned int uint32_t;



typedef unsigned long int uint64_t;
# 66 "/usr/include/stdint.h" 3 4
typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;

typedef long int int_least64_t;






typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;

typedef unsigned long int uint_least64_t;
# 91 "/usr/include/stdint.h" 3 4
typedef signed char int_fast8_t;

typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
# 104 "/usr/include/stdint.h" 3 4
typedef unsigned char uint_fast8_t;

typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
# 123 "/usr/include/stdint.h" 3 4
typedef unsigned long int uintptr_t;
# 135 "/usr/include/stdint.h" 3 4
typedef long int intmax_t;
typedef unsigned long int uintmax_t;
# 4 "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/stdint.h" 2 3 4
# 3 "../common/common.h" 2





struct git_kerb_pipe;

int xread(int fd, uint8_t *buffer, int len);
int xwrite(int fd, const uint8_t *buffer, int len);
int read_net_line(uint8_t *buffer, int *len, int fd);
int write_net_line(const uint8_t *writeout, int len, int fd);
int partial_read(struct git_kerb_pipe *pipe);
int partial_write(struct git_kerb_pipe *pipe);






int pump_data_back_and_forth(int conn, int svc_in, int svc_out,
    pid_t service_pid, krb5_context context,
    krb5_auth_context authcon,
    _Bool sign, _Bool crypt, _Bool kerberized);

int poke_connection_addrs_into_authcon(krb5_context context,
    krb5_auth_context authcon, int socket);




struct git_kerb_pipe {
 krb5_context context;
 krb5_auth_context authcon;
 uint8_t *buffer;
 unsigned int bytes_pending;


 unsigned int offset;
 int infd;
 int outfd;
 _Bool svc;
 _Bool svc_closed;
 _Bool writing;
 _Bool read_length;
 _Bool packmode;
 _Bool sign;
 _Bool encrypt;
 _Bool kerberized;

};
# 4 "gitkrb5-client.h" 2

extern char *target_hostname;
extern char *target_path;
extern int target_portno;

int ingest_url(char *url);
int perform_connect(int *fd);

int get_kerberos_ap_req_and_key(krb5_context context, krb5_auth_context authcon,
     const char *server_principal,
     uint8_t **output, int *outlen,
     int socket);
int ingest_ap_rep(krb5_context context, krb5_auth_context authcon,
     char *buffer, int len);
# 14 "git-remote-gitkrb5.c" 2

int do_capabilities(char *cmd_args);
int do_connect(char *cmd_args);
int do_gitdir(char *cmd_args);
int perform_initial_exchange(int fd);
int handle_server_capabilities(char *string);

char *saved_gitdir_name = ((void *)0);
char *target_hostname = ((void *)0);
char *target_path = ((void *)0);
char *target_service = ((void *)0);
int target_portno = 9418;
char *server_principal = ((void *)0);

struct {
 const char *name;
 int (*handler) (char *cmd_args);
} gitkrb5_commands[] = {
{ "capabilities", do_capabilities },
{ "connect", do_connect },
{ "gitdir", do_gitdir },
{ ((void *)0), ((void *)0) }
};

struct {
 const char *name;
 _Bool wanted;
 _Bool enabled;
 _Bool disable;
} client_capabilities[] = {


{ "krb5-sign", 1, 0, 0 },
{ "krb5-encrypt", 0, 0, 0 },
{ ((void *)0), 0, 0, 0 }
};

int
do_capabilities(char *cmd_args __attribute__((unused)))
{
 const char *name;
 int i;

 for (i = 1; gitkrb5_commands[i].name != ((void *)0); i++) {
  name = gitkrb5_commands[i].name;
  fwrite(name, strlen(name), 1, stdout);
  fwrite("\n", 1, 1, stdout);
 }

 fflush(stdout);
 return 0;
}

int
handle_server_capabilities(char *string)
{
 char *search, *item;
 _Bool mandatory = 0;
 int i, len;

 search = string;
 while ((item = strsep(&search, " ")) != ((void *)0)) {
  mandatory = 0;
  if (*item == '*') {
   item++;
   mandatory = 1;
   if (*item == '\0')
    continue;
  }


  for (i = 0; client_capabilities[i].name != ((void *)0); i++) {
   if (!strcmp(client_capabilities[i].name, item))
    break;
  }


  if (client_capabilities[i].name == ((void *)0)) {

   if (mandatory) {
    fprintf(stderr, "Mandatory server option \"%s\""
      " not supported by client\n",
      item);
    return 1;
   }


   continue;
  }


  if (mandatory && client_capabilities[i].disable) {
   fprintf(stderr, "Mandatory server option \"%s\" "
     "disabled by client configuration\n",
     item);
   return 1;
  }

  if (mandatory)
   client_capabilities[i].enabled = 1;
  else if (client_capabilities[i].wanted)
   client_capabilities[i].enabled = 1;
 }


 string[0] = '\0';
 for (i = 0; client_capabilities[i].name != ((void *)0); i++) {
  if (client_capabilities[i].enabled) {
   strcat(string, client_capabilities[i].name);
   strcat(string, " ");
  }
 }


 len = strlen(string);
 if (string[len-1] == ' ')
  string[len-1] = '\0';


 return 0;
}

int
perform_initial_exchange(int fd)
{
 char *str, *buffer;
 int ret, len;

 buffer = ((void *)0);


 if (target_portno != 9418)
  len = asprintf(&str, "%s-krb5 %s%chost=%s:%d%c", target_service,
     target_path, 0, target_hostname,
     target_portno, 0);
 else
  len = asprintf(&str, "%s-krb5 %s%chost=%s%c", target_service,
     target_path, 0, target_hostname, 0);

 ret = write_net_line((uint8_t*)str, len, fd);
 free(str);
 if (ret != 0) {
  fprintf(stderr, "Couldn't send service-req string: %s\n",
       strerror(ret));
  goto unwind;
 }


 buffer = malloc(0x10000);
 ret = read_net_line((uint8_t *)buffer, &len, fd);
 if (ret != 0) {
  fprintf(stderr, "Couldn't receive server principal: %s\n",
       strerror(ret));
  goto unwind;
 }

 buffer[len] = '\0';
 server_principal = strdup(buffer);


 ret = read_net_line((uint8_t *)buffer, &len, fd);
 if (ret != 0) {
  fprintf(stderr, "Couldn't receive server options: %s\n",
       strerror(ret));
  goto unwind;
 }



 buffer[len] = '\0';
 ret = handle_server_capabilities(buffer);
 if (ret != 0)
  goto unwind;

 len = strlen(buffer);
 ret = write_net_line((uint8_t*)buffer, len, fd);
 if (ret != 0) {
  fprintf(stderr, "Couldn't send capabilities response: %s\n",
       strerror(ret));
  goto unwind;
 }

unwind:
 if (buffer != ((void *)0))
  free(buffer);
 return ret;
}

int
perform_kerberos_exchange(int fd, krb5_context context,
    krb5_auth_context authcon)
{
 uint8_t *ap_req;
 char *buffer;
 int ret, read_len;


 ret = get_kerberos_ap_req_and_key(context, authcon, server_principal,
      &ap_req, &read_len, fd);
 if (ret != 0)
  return ret;

 ret = write_net_line(ap_req, read_len, fd);
 free(ap_req);
 if (ret != 0) {
  fprintf(stderr, "Couldn't pump kerberos ticket at server: %s\n",
        strerror(ret));
  return ret;
 }

 buffer = malloc(0x10000);
 ret = read_net_line((uint8_t *)buffer, &read_len, fd);
 if (ret != 0) {
  free(buffer);
  fprintf(stderr, "Couldn't read AP_REP from server: %s\n",
      error_message(ret));
  return ret;
 }

 ret = ingest_ap_rep(context, authcon, buffer, read_len);
 free(buffer);
 return ret;
}

int
do_connect(char *cmd_args)
{
 krb5_context context;
 krb5_auth_context authcon;
 int ret, fd;

 ret = 1;
 context = ((void *)0);
 authcon = ((void *)0);

 cmd_args += 7;
 while (((*__ctype_b_loc ())[(int) ((*cmd_args))] & (unsigned short int) _ISspace))
  cmd_args++;

 if (*cmd_args == '\0')
  goto unwind;


 cmd_args[strlen(cmd_args) -1] = '\0';
 target_service = strdup(cmd_args);

 ret = krb5_init_context(&context);
 if (ret != 0) {
  fprintf(stderr, "Couldn't init krb5 context: %s\n",
      error_message(ret));
  goto unwind;
 }

 ret = krb5_auth_con_init(context, &authcon);
 if (ret != 0) {
  fprintf(stderr, "Couldn't init kerberos authcon: %s\n",
      error_message(ret));
  goto unwind;
 }

 ret = perform_connect(&fd);
 if (ret > 0) {
  fprintf(stderr, "Couldn't connect to %s: %s\n",
    target_hostname, strerror(ret));
  goto unwind;
 } else if (ret < 0) {

  goto unwind;
 }

 ret = perform_initial_exchange(fd);
 if (ret != 0)
  goto unwind;

 ret = perform_kerberos_exchange(fd, context, authcon);
 if (ret != 0)
  goto unwind;

 fputc('\n', stdout);
 fflush(stdout);

 pump_data_back_and_forth(fd, 0, 1, -1,
    context, authcon,
    client_capabilities[0].enabled,
    client_capabilities[1].enabled,
    1);

unwind:
 if (authcon != ((void *)0))
  krb5_auth_con_free(context, authcon);
 if (context != ((void *)0))
  krb5_free_context(context);



 exit(ret);
}

int
do_gitdir(char *cmd_args)
{
 char *newline;



 cmd_args += 6;
 while (((*__ctype_b_loc ())[(int) ((*cmd_args))] & (unsigned short int) _ISspace))
  cmd_args++;


 newline = strchr(cmd_args, '\n');
 if (newline)
  *newline = '\0';

 if (saved_gitdir_name != ((void *)0))
  free(saved_gitdir_name);

 saved_gitdir_name = strdup(cmd_args);
 fputc('\n', stdout);
 fflush(stdout);
 return 0;
}

int
main(int argc, char **argv)
{
 char buffer[1024];
 struct pollfd fd;
 int ret, len, i;

 if (argc != 3) {
  fprintf(stderr, "Usage: git-remote-gitkrb5 remote_name url\n");
  return 1;
 }

 ingest_url(argv[2]);

 fd.fd = 0;
 while (1) {
  fd.events = 0x001;
  fd.revents = 0;
  ret = poll(&fd, 1, -1);
  if (ret < 0) {
   perror("Polling for commands from git failed");
   exit(1);
  }

  if (fd.revents & (0x008 | 0x010))

   break;

  if (!(fd.revents & 0x001))

   continue;


  fgets(buffer, 1024, stdin);

  for (i = 0; gitkrb5_commands[i].name != ((void *)0); i++) {
   len = strlen(gitkrb5_commands[i].name);

   if (!strncmp(buffer, gitkrb5_commands[i].name, len)) {
    ret = gitkrb5_commands[i].handler(buffer);
    if (ret != 0) {

     exit(1);
    }

    break;
   }
  }


 }

 return 1;
}
