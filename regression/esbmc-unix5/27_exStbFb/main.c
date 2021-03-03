# 1 "exStbFb.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "exStbFb.c"
# 67 "exStbFb.c"
# 1 "/usr/include/unistd.h" 1 3
# 26 "/usr/include/unistd.h" 3
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
# 27 "/usr/include/unistd.h" 2 3


# 173 "/usr/include/unistd.h" 3
# 1 "/usr/include/bits/posix_opt.h" 1 3
# 174 "/usr/include/unistd.h" 2 3
# 188 "/usr/include/unistd.h" 3
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
# 189 "/usr/include/unistd.h" 2 3


typedef __ssize_t ssize_t;





# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 214 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 3 4
typedef unsigned int size_t;
# 198 "/usr/include/unistd.h" 2 3





typedef __gid_t gid_t;




typedef __uid_t uid_t;





typedef __off_t off_t;
# 226 "/usr/include/unistd.h" 3
typedef __useconds_t useconds_t;




typedef __pid_t pid_t;






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
# 1 "/usr/include/getopt.h" 1 3
# 59 "/usr/include/getopt.h" 3
extern char *optarg;
# 73 "/usr/include/getopt.h" 3
extern int optind;




extern int opterr;



extern int optopt;
# 152 "/usr/include/getopt.h" 3
extern int getopt (int ___argc, char *const *___argv, const char *__shortopts)
       __attribute__ ((__nothrow__));
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

# 68 "exStbFb.c" 2
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
# 72 "/usr/include/sys/types.h" 3
typedef __mode_t mode_t;




typedef __nlink_t nlink_t;
# 105 "/usr/include/sys/types.h" 3
typedef __id_t id_t;
# 116 "/usr/include/sys/types.h" 3
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

# 69 "exStbFb.c" 2
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
# 129 "/usr/include/stdint.h" 3
typedef unsigned int uintptr_t;
# 138 "/usr/include/stdint.h" 3
__extension__
typedef long long int intmax_t;
__extension__
typedef unsigned long long int uintmax_t;
# 70 "exStbFb.c" 2
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stdbool.h" 1 3 4
# 71 "exStbFb.c" 2
# 1 "/usr/include/stdio.h" 1 3
# 30 "/usr/include/stdio.h" 3




# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 35 "/usr/include/stdio.h" 2 3
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

# 72 "exStbFb.c" 2
# 1 "/usr/include/fcntl.h" 1 3
# 30 "/usr/include/fcntl.h" 3




# 1 "/usr/include/bits/fcntl.h" 1 3
# 144 "/usr/include/bits/fcntl.h" 3
struct flock
  {
    short int l_type;
    short int l_whence;

    __off_t l_start;
    __off_t l_len;




    __pid_t l_pid;
  };
# 211 "/usr/include/bits/fcntl.h" 3

# 240 "/usr/include/bits/fcntl.h" 3

# 35 "/usr/include/fcntl.h" 2 3
# 76 "/usr/include/fcntl.h" 3
extern int fcntl (int __fd, int __cmd, ...);
# 85 "/usr/include/fcntl.h" 3
extern int open (__const char *__file, int __oflag, ...) __attribute__ ((__nonnull__ (1)));
# 130 "/usr/include/fcntl.h" 3
extern int creat (__const char *__file, __mode_t __mode) __attribute__ ((__nonnull__ (1)));
# 176 "/usr/include/fcntl.h" 3
extern int posix_fadvise (int __fd, __off_t __offset, __off_t __len,
     int __advise) __attribute__ ((__nothrow__));
# 198 "/usr/include/fcntl.h" 3
extern int posix_fallocate (int __fd, __off_t __offset, __off_t __len);
# 220 "/usr/include/fcntl.h" 3

# 73 "exStbFb.c" 2
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

# 74 "exStbFb.c" 2
# 1 "/usr/include/linux/fb.h" 1 3



# 1 "/usr/include/asm/types.h" 1 3





typedef unsigned short umode_t;






typedef __signed__ char __s8;
typedef unsigned char __u8;

typedef __signed__ short __s16;
typedef unsigned short __u16;

typedef __signed__ int __s32;
typedef unsigned int __u32;


typedef __signed__ long long __s64;
typedef unsigned long long __u64;
# 5 "/usr/include/linux/fb.h" 2 3
# 134 "/usr/include/linux/fb.h" 3
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
# 211 "/usr/include/linux/fb.h" 3
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
# 271 "/usr/include/linux/fb.h" 3
enum {

 FB_BLANK_UNBLANK = 0,


 FB_BLANK_NORMAL = 0 + 1,


 FB_BLANK_VSYNC_SUSPEND = 1 + 1,


 FB_BLANK_HSYNC_SUSPEND = 2 + 1,


 FB_BLANK_POWERDOWN = 3 + 1
};
# 298 "/usr/include/linux/fb.h" 3
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
# 352 "/usr/include/linux/fb.h" 3
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
# 75 "exStbFb.c" 2
# 1 "/usr/include/sys/ioctl.h" 1 3
# 24 "/usr/include/sys/ioctl.h" 3



# 1 "/usr/include/bits/ioctls.h" 1 3
# 24 "/usr/include/bits/ioctls.h" 3
# 1 "/usr/include/asm/ioctls.h" 1 3



# 1 "/usr/include/asm/ioctl.h" 1 3
# 1 "/usr/include/asm-generic/ioctl.h" 1 3
# 1 "/usr/include/asm/ioctl.h" 2 3
# 5 "/usr/include/asm/ioctls.h" 2 3
# 25 "/usr/include/bits/ioctls.h" 2 3
# 28 "/usr/include/sys/ioctl.h" 2 3


# 1 "/usr/include/bits/ioctl-types.h" 1 3
# 28 "/usr/include/bits/ioctl-types.h" 3
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
# 31 "/usr/include/sys/ioctl.h" 2 3






# 1 "/usr/include/sys/ttydefaults.h" 1 3
# 38 "/usr/include/sys/ioctl.h" 2 3




extern int ioctl (int __fd, unsigned long int __request, ...) __attribute__ ((__nothrow__));


# 76 "exStbFb.c" 2
# 1 "/usr/include/sys/mman.h" 1 3
# 26 "/usr/include/sys/mman.h" 3
# 1 "/usr/lib/gcc/i386-redhat-linux/4.3.2/include/stddef.h" 1 3 4
# 27 "/usr/include/sys/mman.h" 2 3
# 42 "/usr/include/sys/mman.h" 3
# 1 "/usr/include/bits/mman.h" 1 3
# 43 "/usr/include/sys/mman.h" 2 3





# 58 "/usr/include/sys/mman.h" 3
extern void *mmap (void *__addr, size_t __len, int __prot,
     int __flags, int __fd, __off_t __offset) __attribute__ ((__nothrow__));
# 77 "/usr/include/sys/mman.h" 3
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
# 145 "/usr/include/sys/mman.h" 3
extern int shm_open (__const char *__name, int __oflag, mode_t __mode);


extern int shm_unlink (__const char *__name);


# 77 "exStbFb.c" 2
# 1 "/home/lucas/software/pr11/stb225/src/comps/phStbImage/inc/phStbImage.h" 1
# 64 "/home/lucas/software/pr11/stb225/src/comps/phStbImage/inc/phStbImage.h"
typedef enum {
    phStbImage_frameEvent,
    phStbImage_eofEvent
}phStbImage_eventType;
typedef int (*phStbImage_callback)(phStbImage_eventType event);

typedef struct _setupParams_t
{
      _Bool bScalingEnabled;
      uint32_t *pInBuffer;
      int32_t InBufferHeight;
      int32_t InBufferWidth;
      int32_t OutImageOffsetX;
      int32_t OutImageOffsetY;
      int32_t OutImageHeight;
      int32_t OutImageWidth;
      phStbImage_callback Callback;
} phStbImage_SetupParams_t;


typedef int phStbImageErrorCode_t;


typedef enum {
    phStbBmpFile,
    phStbGifFile,
    phStbPngFile,
    phStbJpegFile,
    phStbVideoDeviceFile,
    phStbUnknownFile } phStbImageFileFormat;
# 122 "/home/lucas/software/pr11/stb225/src/comps/phStbImage/inc/phStbImage.h"
extern phStbImageFileFormat phStbImage_getFileType(const char* filename);
# 133 "/home/lucas/software/pr11/stb225/src/comps/phStbImage/inc/phStbImage.h"
extern phStbImageErrorCode_t phStbImage_GetDimensions( const char* filename, int32_t * pWidth, int32_t * pHeight );
# 144 "/home/lucas/software/pr11/stb225/src/comps/phStbImage/inc/phStbImage.h"
extern phStbImageErrorCode_t phStbImage_Decode( const char* filename, phStbImage_SetupParams_t const * pSetup);
# 78 "exStbFb.c" 2
# 1 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h" 1
# 38 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
# 1 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 1
# 78 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
# 1 "/home/lucas/software/pr11/stb225/build_ctpim/sde2/comps/generated/lib/mipsgnu_linux_el_4KEc/tmFlags.h" 1
# 79 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h" 2
# 149 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
  typedef int8_t Int8;
  typedef int16_t Int16;
  typedef int32_t Int32;
  typedef uint8_t UInt8;
  typedef uint16_t UInt16;
  typedef uint32_t UInt32;






  typedef uint8_t Bool;
# 203 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef char char_t;
# 222 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
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
# 267 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
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
# 350 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef signed long long Int64, *pInt64;
typedef unsigned long long UInt64, *pUInt64;
# 395 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef UInt32 tmErrorCode_t;
typedef UInt32 tmProgressCode_t;


typedef UInt64 tmTimeStamp_t, *ptmTimeStamp_t;





typedef union tmColor3
{
    UBits32 u32;
# 428 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
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
# 472 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
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
# 540 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef Int tmUnitSelect_t, *ptmUnitSelect_t;
# 570 "/home/lucas/software/pr11/stb225/sde2/inc/tmNxTypes.h"
typedef Int tmInstance_t, *ptmInstance_t;


typedef Void (*ptmCallback_t) (UInt32 events, Void *pData, UInt32 userData);
# 39 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h" 2
# 1 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h" 1
# 64 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
typedef struct phStbVideoRenderer_Layer phStbVideoRenderer_Layer_t;


typedef enum phStbVideoRenderer_Layer_InputFormat
{

    phStbVideoRenderer_Layer_InputFormatNone = 0,

    phStbVideoRenderer_Layer_InputFormatSD,

    phStbVideoRenderer_Layer_InputFormatHD,

    phStbVideoRenderer_Layer_InputFormatCount
} phStbVideoRenderer_Layer_InputFormat_t;
# 100 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetClockId(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pClockId);
# 116 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_BlankOutput(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool blank);
# 132 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_ShowLayer(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool show);
# 147 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetBrightness(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t brightness);
# 162 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetBrightness(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pBrightness);
# 177 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetContrast(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t contrast);
# 192 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetContrast(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pContrast);
# 207 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetSaturation(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t saturation);
# 222 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetSaturation(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pSaturation);
# 237 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetHue(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t hue);
# 252 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetHue(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pHue);
# 269 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetWhitePoint(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t red,
    uint32_t green,
    uint32_t blue);
# 288 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetWhitePoint(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pRed,
    uint32_t *pGreen,
    uint32_t *pBlue);
# 320 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetSrcRectangle(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t ulx,
    uint32_t uly,
    uint32_t lrx,
    uint32_t lry,
    _Bool immediate);
# 345 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetSrcRectangle(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pUlx,
    uint32_t *pUly,
    uint32_t *pLrx,
    uint32_t *pLry);
# 376 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetDestRectangle(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t ulx,
    uint32_t uly,
    uint32_t lrx,
    uint32_t lry);
# 400 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetDestRectangle(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pUlx,
    uint32_t *pUly,
    uint32_t *pLrx,
    uint32_t *pLry);
# 424 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetAlphaLevel(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t alphaLevel);
# 439 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetAlphaLevel(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t *pAlpha);
# 452 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetFlickerFiltering(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool enable);
# 465 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetFlickerFiltering(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool *enabled);
# 515 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_IsScalingSupported(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool *pScalingSupported);
# 531 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetInput(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t numPackets,
    phStbVideoRenderer_Layer_InputFormat_t format);
# 550 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_SetBufferPointer(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t stride,
    uint32_t height,
    void *pBuffer);
# 569 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_Pause(
    phStbVideoRenderer_Layer_t *pInstance,
    _Bool pause);
# 586 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetCurrentFrameInfo(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t* pTimeCode,
    uint32_t* pPictureType,
    uint32_t* pTemporalReference);
# 605 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer_Layer.h"
extern tmErrorCode_t phStbVideoRenderer_Layer_GetVideoClockTime(
    phStbVideoRenderer_Layer_t *pInstance,
    uint32_t* pTimeLo,
    uint32_t* pTimeHigh);
# 40 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h" 2
# 85 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
typedef struct phStbVideoRenderer phStbVideoRenderer_t;



typedef enum phStbVideoRenderer_Planes
{
    phStbVideoRenderer_Planes_First = 0,
    phStbVideoRenderer_Planes_Second = 1,
    phStbVideoRenderer_Planes_Third = 2,
    phStbVideoRenderer_Planes_Fourth = 3,
    phStbVideoRenderer_Planes_Bottom = phStbVideoRenderer_Planes_Fourth,
    phStbVideoRenderer_Planes_Top = phStbVideoRenderer_Planes_First,
    phStbVideoRenderer_Planes_Count = 4
} phStbVideoRenderer_Planes_t;



typedef enum phStbVideoRenderer_Layers
{
    phStbVideoRenderer_Layers_Display1 = 0,
    phStbVideoRenderer_Layers_Display2 = 1,
    phStbVideoRenderer_Layers_Graphics1 = 2,
    phStbVideoRenderer_Layers_Graphics2 = 3,
    phStbVideoRenderer_Layers_None = 5
} phStbVideoRenderer_Layers_t;



typedef struct phStbVideoRenderer_Plane
{
    phStbVideoRenderer_Layers_t layer;
    _Bool enabled;
} phStbVideoRenderer_Plane_t;







typedef enum phStbVideoRenderer_OutputPath
{
    phStbVideoRenderer_OutputPath_HdDac = 0,
    phStbVideoRenderer_OutputPath_SdDac = 1,
    phStbVideoRenderer_OutputPath_Dvo = 2,
    phStbVideoRenderer_OutputPath_Count = 3
} phStbVideoRenderer_OutputPath_t;







typedef enum phStbVideoRenderer_DisplayChain
{
    phStbVideoRenderer_DisplayChain_Downscaled = 0,
    phStbVideoRenderer_DisplayChain_Main = 1,
    phStbVideoRenderer_DisplayChain_Count = 2
} phStbVideoRenderer_DisplayChain_t;


typedef enum phStbVideoRenderer_FrameRate
{
    phStbVideoRenderer_FrameRateNone = 0x00,
    phStbVideoRenderer_FrameRate_23_976 = 0x01,
    phStbVideoRenderer_FrameRate_24 = 0x02,
    phStbVideoRenderer_FrameRate_25 = 0x03,
    phStbVideoRenderer_FrameRate_29_97 = 0x04,
    phStbVideoRenderer_FrameRate_30 = 0x05,
    phStbVideoRenderer_FrameRate_50 = 0x06,
    phStbVideoRenderer_FrameRate_59_94 = 0x07,
    phStbVideoRenderer_FrameRate_60 = 0x08
} phStbVideoRenderer_FrameRate_t;






typedef enum phStbVideoRenderer_NotificationTypes
{

    phStbVideoRenderer_NotificationTypes_None = 0,


    phStbVideoRenderer_NotificationTypes_EndOfStream = (1 << 0),


    phStbVideoRenderer_NotificationTypes_FormatChange = (1 << 1),



    phStbVideoRenderer_NotificationTypes_VidOut = (1 << 2),





    phStbVideoRenderer_NotificationTypes_Sync = (1 << 3),



    phStbVideoRenderer_NotificationTypes_FrameRepeated = (1 << 4),



    phStbVideoRenderer_NotificationTypes_FirstFramePresented = (1 << 5),


    phStbVideoRenderer_NotificationTypes_Guard = (1 << 6)

} phStbVideoRenderer_NotificationTypes_t;



typedef enum phStbVideoRenderer_VbiPosition
{
    phStbVideoRenderer_VbiPosition_Start = 0,
    phStbVideoRenderer_VbiPosition_End = 1

}phStbVideoRenderer_VbiPosition_t;


typedef enum phStbVideoRenderer_Parity
{
    phStbVideoRenderer_Parity_Even = 0,
    phStbVideoRenderer_Parity_Odd = 1
} phStbVideoRenderer_Parity_t;
# 234 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetInstance(
    tmUnitSelect_t unitNum,
    phStbVideoRenderer_t **ppRendInst);
# 251 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetVideoLayer(
    phStbVideoRenderer_Layers_t layerSelect,
    phStbVideoRenderer_Layer_t **ppLayer,
    char *pLayerName);
# 276 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_SelectOutput(
    phStbVideoRenderer_OutputPath_t outputPath,
    tmUnitSelect_t unitNum,
    phStbVideoRenderer_DisplayChain_t displayChain);
# 295 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_SetOutputFormat(
    phStbVideoRenderer_DisplayChain_t displayChain,
    const char *pVideoFormat);
# 311 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetOutputFormat(
    phStbVideoRenderer_DisplayChain_t displayChain,
    char *pVideoFormat);
# 324 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_SetAnalogSlave(
    const _Bool *pSlave);
# 337 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetOutputResolution(
    phStbVideoRenderer_DisplayChain_t displayChain,
    uint32_t *pWidth,
    uint32_t *pHeight);
# 354 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetOutputFrameRate(
    phStbVideoRenderer_DisplayChain_t displayChain,
    phStbVideoRenderer_FrameRate_t *pFrameRate);
# 367 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetOutputScanType(
    phStbVideoRenderer_DisplayChain_t displayChain,
    _Bool *pInterlaced);
# 389 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_SetPlaneStack(
    phStbVideoRenderer_t *pInstance,
    const phStbVideoRenderer_Plane_t *pLayers);
# 411 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetPlaneStack(
    phStbVideoRenderer_t *pInstance,
    phStbVideoRenderer_Plane_t *pLayers);
# 427 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_BlankOutput(
    phStbVideoRenderer_t *pInstance,
    _Bool blank);
# 446 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_SetBackgroundColour(
    phStbVideoRenderer_t *pInstance,
    uint32_t red,
    uint32_t green,
    uint32_t blue);
# 467 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetBackgroundColour(
    phStbVideoRenderer_t *pInstance,
    uint32_t *pRed,
    uint32_t *pGreen,
    uint32_t *pBlue);
# 483 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_RegisterVBICallback(
    ptmCallback_t callback,
    phStbVideoRenderer_VbiPosition_t vbiPosition,
    phStbVideoRenderer_DisplayChain_t displayChain);
# 497 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_EnableVBI(
    phStbVideoRenderer_VbiPosition_t vbiPosition,
    phStbVideoRenderer_DisplayChain_t displayChain);
# 510 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_DisableVBI(
    phStbVideoRenderer_VbiPosition_t vbiPosition,
    phStbVideoRenderer_DisplayChain_t displayChain);
# 524 "/home/lucas/software/pr11/stb225/src/streamingSystem/comps/phStbVideoRenderer/inc/phStbVideoRenderer.h"
extern tmErrorCode_t phStbVideoRenderer_GetFieldLine(
    phStbVideoRenderer_Parity_t *parity,
    unsigned int *line,
    phStbVideoRenderer_DisplayChain_t displayChain);
# 79 "exStbFb.c" 2
static struct fb_var_screeninfo vinfo;
static struct fb_fix_screeninfo finfo;
static uint8_t *fbbase = 0;
static _Bool doLoop = 1;
# 96 "exStbFb.c"
static char* printErrorCode(phStbImageErrorCode_t errorCode)
{

    switch(errorCode)
    {
        case(0):
            return "No Error";
        case(-1):
            return "Wrong file format";
        case(-2):
            return "Unable to open file";
        case(-3):
            return "File truncated";
        case(-4):
            return "Out of memory";
        case(-5):
            return "Null parameter";
        case(-6):
            return "Unable to Memory Map";
        case(-7):
            return "Null memory pointer";
        case(-8):
            return "File not closed";
        default:
            return "Unrecognised error code";
    }

}

static void rect(int32_t x, int32_t y,
                 int32_t width, int32_t height,
                 uint32_t r, uint32_t g, uint32_t b)
{

    int32_t xstart = x;
    int32_t ystart = y;

    for (y = ystart; y < ystart + height; y++ ) {
        for (x = xstart; x < xstart + width; x++ ) {
            int32_t offset =
                ((x+(int32_t)vinfo.xoffset) * ((int32_t)vinfo.bits_per_pixel/8)) +
                ((y+(int32_t)vinfo.yoffset) * (int32_t)finfo.line_length);

            if ( vinfo.bits_per_pixel == 32 ) {
                *(fbbase + offset + 0) = 255;
                *(fbbase + offset + 1) = (uint8_t)r;
                *(fbbase + offset + 2) = (uint8_t)g;
                *(fbbase + offset + 3) = (uint8_t)b;
            } else {
                uint16_t t = (uint16_t)r<<11 | (uint16_t)g << 5 | (uint16_t)b;
                *((uint16_t *)(fbbase + offset)) = t;
            }
        }
    }

}

typedef struct _DFBRectangle
{
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
}DFBRectangle;


static void clearDifference(DFBRectangle const *pActualRect)
{

    DFBRectangle top, bottom, left, right;

    if ((vinfo.xres != (uint32_t)pActualRect->w) ||
        (vinfo.yres != (uint32_t)pActualRect->h))
    {
        top.x = 0;
        top.y = 0;
        top.w = 0;
        top.h = 0;
        bottom = left = right = top;

        if (0 != pActualRect->y)
        {
            top.y = 0;
            top.x = 0;
            top.w = (int32_t)vinfo.xres;
            top.h = pActualRect->y;
        }

        if (0 != pActualRect->x)
        {
            left.y = 0;
            left.x = 0;
            left.w = pActualRect->x;
            left.h = (int32_t)vinfo.yres;
        }

        if (vinfo.yres != (uint32_t)(pActualRect->y + pActualRect->h))
        {
            bottom.x = 0;
            bottom.y = pActualRect->y + pActualRect->h;
            bottom.w = (int32_t)vinfo.xres;
            bottom.h = (int32_t)vinfo.yres - bottom.y;
        }
        if (vinfo.xres != (uint32_t)(pActualRect->x + pActualRect->w))
        {
            right.x = pActualRect->x + pActualRect->w;
            right.y = 0;
            right.w = (int32_t)vinfo.xres - right.x;
            right.h = (int32_t)vinfo.yres;
        }

        if ((top.w != 0) && (top.h != 0))
        {
            rect(top.x, top.y, top.w, top.h, 0, 0, 0);
        }
        if ((bottom.w != 0) && (bottom.h != 0))
        {
            rect(bottom.x, bottom.y, bottom.w, bottom.h, 0, 0, 0);
        }
        if ((left.w != 0) && (left.h != 0))
        {
            rect(left.x, left.y, left.w, left.h, 0, 0, 0);
        }
        if ((right.w != 0) && (right.h != 0))
        {
            rect(right.x, right.y, right.w, right.h, 0, 0, 0);
        }
    }

}
# 241 "exStbFb.c"
static void RGB2YUV(uint32_t value, uint8_t *pY, uint8_t *pU, uint8_t *pV)
{

    int32_t blue;
    int32_t green;
    int32_t red;

    red = (int32_t)(((value >> 16) & 0xFF));
    green = (int32_t)(((value >> 8) & 0xFF));
    blue = (int32_t)((value & 0xFF));

    if (pY)
    {
        *pY = (uint8_t)((((uint32_t)(((47)*red) + ((157)*green) + ((16)*blue))) >> 8) + (16));
    }

    if (pU)
    {
        *pU = (uint8_t)((((uint32_t)(((-26)*red) + ((-86)*green) + ((112)*blue))) >> 8) + (128));
    }

    if (pV)
    {
        *pV = (uint8_t)((((uint32_t)(((112)*red) + ((-102)*green) + ((-10)*blue))) >> 8) + (128));
    }

}

static int32_t gifCallback(phStbImage_eventType event)
{

    static int32_t imageCount = 0;

    if (event == phStbImage_eofEvent)
    {

        return ((imageCount != 0) && doLoop) ? 0 : -1;
    }

    imageCount++;
    return 0;

}

int32_t main(int32_t argc, char *argv[])
{
    int32_t fd = 0;
    int32_t i;
    char *path = "/dev/fb0";
    phStbImageErrorCode_t errorCode;
    phStbImage_SetupParams_t params;
    int32_t width;
    int32_t height;
    int32_t out_width = 0;
    int32_t out_height = 0;
    float heightR;
    float widthR;
    float scale;
    int32_t modifiedHeight;
    _Bool doScale = 1;
    _Bool doAnimate = 1;
    _Bool doClear = 1;
    _Bool squarepixels = 0;
    int32_t setOrigin = 0;
    int32_t originX = 0;
    int32_t originY = 0;
    int32_t increment = 1;
    uint32_t screensize = 0;
    int32_t error;


    DFBRectangle actualRect;


__ESBMC_assume(argc>=0 && argc<(sizeof(argv)/sizeof(char)));
int counter;
for(counter=0; counter<argc; counter++)
  __ESBMC_assume(argv[counter]!=((void *)0));



    if (argc < 2)
    {
       (void)printf("Usage : %s <filename> [-fb </dev/fb[0|1|2]|/dev/vrend/display[1|2]>] [-width <width>] [-height <height>] [-origin x,y] [-noscale] [-noanimate] [-noclear] [-noloop] [-squarepixels]\n", argv[0]);
       exit(0);
    }

    (void)printf("============================================================================================\n");


    for(i=2; i<argc; i+=increment)
    {
       increment = 1;
       if (!strcmp(argv[i], "-fb"))
       {
           if ((i+1)<argc)
           {
               path = argv[i+1];
           }
           else
           {
               (void)printf("Error: Insufficient parameters for '-fb'\n");
               exit(1);
           }
           increment++;
       }
       if (!strcmp(argv[i], "-width"))
       {

           if ((i+1)<argc)
           {
               out_width = atoi(argv[i+1]);
           }
           else
           {
               (void)printf("Error: Insufficient parameters for '-width'\n");
               exit(1);
           }
           increment++;
       }

       if (!strcmp(argv[i], "-height"))
       {
           if ((i+1)<argc)
           {
               out_height = atoi(argv[i+1]);
           }
           else
           {
               (void)printf("Error: Insufficient parameters for '-height'\n");
               exit(1);
           }
           increment++;
       }
       if (!strcmp(argv[i], "-noscale"))
       {
           doScale = 0;
       }
       if (!strcmp(argv[i], "-noanimate"))
       {
           doAnimate = 0;
       }
       if (!strcmp(argv[i], "-noclear"))
       {
           doClear = 0;
       }
       if (!strcmp(argv[i], "-noloop"))
       {
           doLoop = 0;
       }
       if (!strcmp(argv[i], "-squarepixels"))
       {
           squarepixels = 1;
       }
       if (!strcmp(argv[i], "-origin"))
       {
           if ((i+1)<argc)
           {
               setOrigin = 1;
               (void)sscanf(argv[i+1], "%d,%d", &originX, &originY);
           }
           else
           {
               (void)printf("Error: Insufficient parameters for '-origin'\n");
               exit(1);
           }
           increment++;
       }
    }
    if (strstr(path, "/dev/fb"))
    {


        fd = open(path, 02);
        if (fd < 0) {
            (void)printf("Error: cannot open framebuffer device %s\n", path);
            perror(path);
            exit(1);
        }


        error = ioctl(fd, 0x4602, &finfo);
        if (error!=0) {
            (void)printf("Error reading fixed information.\n");
            perror(path);
            exit(2);
        }


        if (ioctl(fd, 0x4600, &vinfo)) {
            (void)printf("Error reading variable information.\n");
            perror(path);
            exit(3);
        }


        screensize = (uint32_t)(vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8);


        fbbase = (uint8_t *)mmap(0, screensize, 0x1 | 0x2, 0x01,
                                       fd, 0);
        if ((int32_t)fbbase == -1) {
            (void)printf("Error: failed to map framebuffer device to memory.\n");
            perror(path);
            exit(4);
        }


        if (out_width == 0)
        {
            out_width = (int32_t)vinfo.xres;
        }
        if (out_height == 0)
        {
            out_height = (int32_t)vinfo.yres;
        }


        errorCode = phStbImage_GetDimensions( argv[1], &width, &height );
        if (errorCode == 0)
        {
           (void)printf("Image dimensions for '%s' are %d x %d\n", argv[1], width, height);
        }
        else
        {
           (void)printf("Failed to get image dimensions for '%s' - error code '%s'\n", argv[1], printErrorCode(errorCode));
           exit(0);
        }

        if ((originX >= 0) && (originX < (int32_t)vinfo.xres) &&
            (originY >= 0) && (originY < (int32_t)vinfo.yres))
        {
            actualRect.x = originX;
            actualRect.y = originY;
        }
        else
        {
            actualRect.x = 0;
            actualRect.y = 0;
        }
        if (doScale)
        {
            if (!squarepixels)
            {

                modifiedHeight = (int32_t)((float)height*((16.0/720.0) / (9.0/576)));
                (void)printf("%s Dimensions: %dx%d (non-square pixels -> %dx%d)\n", argv[1], width, height, width, modifiedHeight);
            }
            else
            {
                modifiedHeight = height;
            }





   if ((float)width != 0)
    widthR = (float)out_width / (float)width;
   if ((float)modifiedHeight != 0)
             heightR = (float)out_height / (float)modifiedHeight;

            if (heightR > widthR)
            {
               scale = widthR;
            }
            else
            {
               scale = heightR;
            }

            (void)printf("Scale Factor is %f\n", scale);

            actualRect.w = ((int32_t)((float)width * scale)/2)*2;
            actualRect.h = ((int32_t)((float)modifiedHeight * scale)/2)*2;
        }
        else
        {
            actualRect.w = width;
            actualRect.h = height;
        }
        if (setOrigin==0)
        {

            if (actualRect.w < (int32_t)vinfo.xres)
            {
               actualRect.x = (((int32_t)vinfo.xres - actualRect.w) / 4)*2;
            }
            if (actualRect.h < (int32_t)vinfo.yres)
            {
               actualRect.y = (((int32_t)vinfo.yres - actualRect.h) / 4)*2;
            }
        }

        (void)printf("Rendering to %d,%d %dx%d\n", actualRect.x, actualRect.y, actualRect.w, actualRect.h);


        params.bScalingEnabled = doScale;
        params.pInBuffer = (uint32_t*)(fbbase);
        params.InBufferHeight = (int32_t)vinfo.yres;
        params.InBufferWidth = (int32_t)vinfo.xres;
        params.OutImageOffsetX = actualRect.x;
        params.OutImageOffsetY = actualRect.y;
        params.OutImageHeight = actualRect.h;
        params.OutImageWidth = actualRect.w;
        params.Callback = doAnimate ? gifCallback : ((void *)0);


        if (doClear)
        {
            clearDifference(&actualRect);
        }


        errorCode = phStbImage_Decode( argv[1], &params);
        if (errorCode == 0)
        {
           (void)printf("Decoded Image '%s'\n", argv[1]);
        }
        else
        {
           (void)printf("Failed to decode image '%s' - error code '%s'\n", argv[1], printErrorCode(errorCode));
        }


        (void)munmap(fbbase, screensize);
        (void)close(fd);
    }
    else
    {
        uint32_t dims[4];
        uint32_t* decodeBuffer;



        fd = open(path, 02);
        if (fd < 0) {
            (void)printf("Error: cannot open video device %s\n", path);
            perror(path);
            exit(1);
        }


        error = ioctl(fd, (((2U) << (((0 +8)+8)+14)) | ((('V')) << (0 +8)) | (((1)) << 0) | ((((sizeof(uint32_t*)))) << ((0 +8)+8))), dims);
        if (error != 0) {
            (void)printf("Error reading vidoe layer information.\n");
            perror(path);
            exit(2);
        }


        out_width = (int32_t)(dims[2] - dims[0]);
        out_height = (int32_t)(dims[3] - dims[1]);


        decodeBuffer = (uint32_t*) malloc((size_t)(out_width*out_height*(int32_t)sizeof(uint32_t)));

        if (decodeBuffer)
        {
            (void)printf("Rendering to %d,%d %dx%d\n", 0, 0, out_width, out_height);

            params.bScalingEnabled = 1;
            params.pInBuffer = decodeBuffer;
            params.InBufferHeight = out_height;
            params.InBufferWidth = out_width;
            params.OutImageOffsetX = 0;
            params.OutImageOffsetY = 0;
            params.OutImageHeight = out_height;
            params.OutImageWidth = out_width;
            params.Callback = ((void *)0);

            errorCode = phStbImage_Decode( argv[1], &params);
            if (errorCode == 0)
            {
               (void)printf("Decoded Image '%s'\n", argv[1]);
            }
            else
            {
               (void)printf("Failed to decode image '%s' - error code '%s'\n", argv[1], printErrorCode(errorCode));
            }

            if (errorCode == 0)
            {
                int32_t row;
                int32_t col;
                uint32_t* pSrcBuffer;
                uint8_t* YUVBuffer;
                uint8_t* pDestBuffer;
                pSrcBuffer = decodeBuffer;


                YUVBuffer = (uint8_t*) malloc((size_t)((((out_width * out_height * 2) / 4096) + 1) * 4096));
                if (YUVBuffer)
                {
                    int32_t length;
                    pDestBuffer = YUVBuffer;
                    for(row=0; row<out_height; row++)
                    {
                        for(col=0; col<out_width; col++)
                        {
                            uint8_t Y;
                            uint8_t U;
                            uint8_t V;

                             RGB2YUV(*pSrcBuffer, &Y, &U, &V);
                            if (col%2 == 1)
                            {

                                *pDestBuffer = V;
                                *(pDestBuffer+1) = Y;
                            }
                            else
                            {

                                *pDestBuffer = U;
                                *(pDestBuffer+1) = Y;
                            }
                            pSrcBuffer++;
                            pDestBuffer+=2;
                        }
                    }


                    pDestBuffer = YUVBuffer;
                    do
                    {
                        length = write(fd, pDestBuffer, 4096);
                        if (length <0)
                        {
                            (void)printf("Failed at location %d\n", pDestBuffer-YUVBuffer);
                        }
                        pDestBuffer += length;
                    }while(((pDestBuffer-YUVBuffer) < (out_width * out_height * 2)) && (length>0));


                    free(YUVBuffer);
                }
                else
                {
                    (void)printf("Failed to allocate YUV Buffer\n");
                }
            }


            free(decodeBuffer);
        }
        else
        {
            (void)printf("Failed to allocate decode buffer\n");
        }

        (void)close(fd);

    }
    return 0;
}
