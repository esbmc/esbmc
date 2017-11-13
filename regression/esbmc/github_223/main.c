






typedef long unsigned int size_t;
typedef int wchar_t;








typedef enum
{
  P_ALL,
  P_PID,
  P_PGID
} idtype_t;


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


typedef long int __fsword_t;

typedef long int __ssize_t;


typedef long int __syscall_slong_t;

typedef unsigned long int __syscall_ulong_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;






static __inline unsigned int
__bswap_32 (unsigned int __bsx)
{
  return __builtin_bswap32 (__bsx);
}
static __inline __uint64_t
__bswap_64 (__uint64_t __bsx)
{
  return __builtin_bswap64 (__bsx);
}

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
typedef union
  {
    union wait *__uptr;
    int *__iptr;
  } __WAIT_STATUS __attribute__ ((__transparent_union__));


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


extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__ , __leaf__)) ;




extern double atof (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





__extension__ extern long long int atoll (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern float strtof (const char *__restrict __nptr,
       char **__restrict __endptr) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

extern long double strtold (const char *__restrict __nptr,
       char **__restrict __endptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));




__extension__
extern long long int strtoq (const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





__extension__
extern long long int strtoll (const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

extern char *l64a (long int __n) __attribute__ ((__nothrow__ , __leaf__)) ;


extern long int a64l (const char *__s)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;










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


typedef __clock_t clock_t;





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
    __syscall_slong_t tv_nsec;
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






    __fd_mask __fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];


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
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
__extension__
extern unsigned int gnu_dev_minor (unsigned long long int __dev)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
__extension__
extern unsigned long long int gnu_dev_makedev (unsigned int __major,
            unsigned int __minor)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));






typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
typedef unsigned long int pthread_t;


union pthread_attr_t
{
  char __size[56];
  long int __align;
};

typedef union pthread_attr_t pthread_attr_t;





typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
typedef union
{
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









extern long int random (void) __attribute__ ((__nothrow__ , __leaf__));


extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__ , __leaf__));





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));







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
       int32_t *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));






extern int rand (void) __attribute__ ((__nothrow__ , __leaf__));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__ , __leaf__));




extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__ , __leaf__));







extern double drand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern long int nrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern long int jrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) __attribute__ ((__nothrow__ , __leaf__));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    __extension__ unsigned long long int __a;

  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));









extern void *malloc (size_t __size) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) ;

extern void *calloc (size_t __nmemb, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) ;










extern void *realloc (void *__ptr, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) __attribute__ ((__nothrow__ , __leaf__));




extern void cfree (void *__ptr) __attribute__ ((__nothrow__ , __leaf__));










extern void *alloca (size_t __size) __attribute__ ((__nothrow__ , __leaf__));











extern void *valloc (size_t __size) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) ;


extern void abort (void) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));













extern void _Exit (int __status) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));






extern char *getenv (const char *__name) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) ;

extern int putenv (char *__string) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (const char *__name, const char *__value, int __replace)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (const char *__name) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) __attribute__ ((__nothrow__ , __leaf__));
extern char *mktemp (char *__template) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) ;





extern int system (const char *__command) ;

extern char *realpath (const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__ , __leaf__)) ;






typedef int (*__compar_fn_t) (const void *, const void *);



extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;







extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
extern int abs (int __x) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;



__extension__ extern long long int llabs (long long int __x)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;







extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;




__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) ;

extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));






extern int mblen (const char *__s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));


extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));


extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__ , __leaf__));



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n)
     __attribute__ ((__nothrow__ , __leaf__));








extern int rpmatch (const char *__response) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) ;
extern int getsubopt (char **__restrict __optionp,
        char *const *__restrict __tokens,
        char **__restrict __valuep)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2, 3))) ;
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));




void kfree(void*);

int __VERIFIER_nondet_int(void);
void *__VERIFIER_nondet_pointer(void);

int ldv_nonpositive(void) {
 int r = __VERIFIER_nondet_int();
 if(r<0) return r;
 else return 0;
}

int ldv_positive(void) {
 int r = __VERIFIER_nondet_int();
 if(r>0) return r;
 else return 1;
}

void *memcpy(void*, const void *, size_t);
void *memset(void*, int, size_t);

void *ldv_malloc(size_t size) {
 if(__VERIFIER_nondet_int()) {
  return malloc(size);
 } else {
  return 0;
 }
};

void *ldv_zalloc(size_t size) {
 return calloc(size, 1);
}





struct ldv_list_head {
 struct ldv_list_head *next, *prev;
};






static inline void LDV_INIT_LIST_HEAD(struct ldv_list_head *list)
{
 list->next = list;
 list->prev = list;
}

static inline void __ldv_list_add(struct ldv_list_head *new,
                            struct ldv_list_head *prev,
                            struct ldv_list_head *next)
{
 next->prev = new;
 new->next = next;
 new->prev = prev;
 prev->next = new;
}

static inline void __ldv_list_del(struct ldv_list_head * prev, struct ldv_list_head * next)
{
 next->prev = prev;
 prev->next = next;
}

static inline void ldv_list_add(struct ldv_list_head *new, struct ldv_list_head *head)
{
 __ldv_list_add(new, head, head->next);
}

static inline void ldv_list_add_tail(struct ldv_list_head *new, struct ldv_list_head *head)
{
 __ldv_list_add(new, head->prev, head);
}

static inline void ldv_list_del(struct ldv_list_head *entry)
{
 __ldv_list_del(entry->prev, entry->next);


}
struct ldv_list_head ldv_global_msg_list = { &(ldv_global_msg_list), &(ldv_global_msg_list) };

struct ldv_msg {
 void *data;
 struct ldv_list_head list;
};

struct ldv_msg *ldv_msg_alloc() {
 struct ldv_msg *msg;
 msg = (struct ldv_msg*)ldv_malloc(sizeof(struct ldv_msg));
 if(msg) {
  msg->data=0;
  LDV_INIT_LIST_HEAD(&msg->list);
 }
 return msg;
}

int ldv_msg_fill(struct ldv_msg *msg, void *buf, int len) {
 void *data;
 data = ldv_malloc(len);
 if(!data) return - -3;
 memcpy(data, buf, len);
 msg->data = data;
 return 0;
}

void ldv_msg_free(struct ldv_msg *msg) {
 if(msg) {
  free(msg->data);
  free(msg);
 }
}

int ldv_submit_msg(struct ldv_msg *msg) {
  if(__VERIFIER_nondet_int()) {
   ldv_list_add(&msg->list, &ldv_global_msg_list);
   return 0;
  }
  return -1;
}

void ldv_destroy_msgs(void) {
 struct ldv_msg *msg;
 struct ldv_msg *n;
 for (msg = ({ const typeof( ((typeof(*msg) *)0)->list ) *__mptr = ((&ldv_global_msg_list)->next); (typeof(*msg) *)( (char *)__mptr - ((size_t) &((typeof(*msg) *)0)->list) );}), n = ({ const typeof( ((typeof(*(msg)) *)0)->list ) *__mptr = ((msg)->list.next); (typeof(*(msg)) *)( (char *)__mptr - ((size_t) &((typeof(*(msg)) *)0)->list) );}); &msg->list != (&ldv_global_msg_list); msg = n, n = ({ const typeof( ((typeof(*(n)) *)0)->list ) *__mptr = ((n)->list.next); (typeof(*(n)) *)( (char *)__mptr - ((size_t) &((typeof(*(n)) *)0)->list) );})) {
  ldv_list_del(&msg->list);

  ldv_msg_free(msg);
 }
}
struct ldv_device {
 void *platform_data;
 void *driver_data;
 struct ldv_device *parent;
};

static inline void *ldv_dev_get_drvdata(const struct ldv_device *dev)
{
 return dev->driver_data;
}

static inline void ldv_dev_set_drvdata(struct ldv_device *dev, void *data)
{
 dev->driver_data = data;
}






struct ldv_usb_interface_descriptor {
 char bLength;
 char bDescriptorType;
 char bInterfaceNumber;
 char bAlternateSetting;
 char bNumEndpoints;
 char bInterfaceClass;
 char bInterfaceSubClass;
 char bInterfaceProtocol;
 char iInterface;
} __attribute__ ((packed));

struct ldv_usb_host_interface {
 struct ldv_usb_interface_descriptor desc;
};

struct ldv_usb_interface {


 struct ldv_usb_host_interface *altsetting;
 struct ldv_usb_host_interface *cur_altsetting;
 struct ldv_device dev;
};






typedef struct {
        int counter;
} ldv_atomic_t;

struct ldv_kref {
        ldv_atomic_t refcount;
};

struct ldv_kobject {
        char *name;
        struct ldv_list_head entry;




        struct ldv_kref kref;
};

static inline int ldv_atomic_add_return(int i, ldv_atomic_t *v)
{
        int temp;
        temp = v->counter;
        temp += i;
        v->counter = temp;
        return temp;
}

static inline int ldv_atomic_sub_return(int i, ldv_atomic_t *v)
{
        int temp;
        temp = v->counter;
        temp -= i;
        v->counter = temp;
        return temp;
}






static inline int ldv_kref_sub(struct ldv_kref *kref, unsigned int count,
            void (*release)(struct ldv_kref *kref))
{


        if ((ldv_atomic_sub_return(((int) count), (&kref->refcount)) == 0)) {
                release(kref);
                return 1;
        }
        return 0;
}





static inline void ldv_kref_init(struct ldv_kref *kref)
{
        (((&kref->refcount)->counter) = (1));
}





static inline void ldv_kref_get(struct ldv_kref *kref)
{





        ldv_atomic_add_return(1, (&kref->refcount));
}

static inline int ldv_kref_put(struct ldv_kref *kref, void (*release)(struct ldv_kref *kref))
{
        return ldv_kref_sub(kref, 1, release);
}






void ldv_kobject_del(struct ldv_kobject *kobj)
{


        if (!kobj)
                return;







}

static void ldv_kobject_cleanup(struct ldv_kobject *kobj)
{

        char *name = kobj->name;
        free(kobj);


        if (name) {
                free(name);
        }
}

static void ldv_kobject_release(struct ldv_kref *kref) {
 struct ldv_kobject *kobj = ({ const typeof( ((struct ldv_kobject *)0)->kref ) *__mptr = (kref); (struct ldv_kobject *)( (char *)__mptr - ((size_t) &((struct ldv_kobject *)0)->kref) );});
        ldv_kobject_cleanup(kobj);
}







void ldv_kobject_put(struct ldv_kobject *kobj)
{
        if (kobj) {

                ldv_kref_put(&kobj->kref, ldv_kobject_release);
        }
}





struct ldv_kobject *ldv_kobject_get(struct ldv_kobject *kobj)
{
        if (kobj)
                ldv_kref_get(&kobj->kref);
        return kobj;
}

static void ldv_kobject_init_internal(struct ldv_kobject *kobj)
{
        if (!kobj)
                return;
        ldv_kref_init(&kobj->kref);
        LDV_INIT_LIST_HEAD(&kobj->entry);




}

void ldv_kobject_init(struct ldv_kobject *kobj)
{


        if (!kobj) {

                goto error;
        }
        ldv_kobject_init_internal(kobj);

        return;
error:
 return;
}
struct ldv_kobject *ldv_kobject_create(void)
{
        struct ldv_kobject *kobj;

        kobj = ldv_malloc(sizeof(*kobj));
        if (!kobj)
                return 0;
 memset(kobj, 0, sizeof(*kobj));

        ldv_kobject_init(kobj);
        return kobj;
}





struct A {
 void *p;
};

int f(void) {
 return __VERIFIER_nondet_int();
}

int g(void) {
 return __VERIFIER_nondet_int();
}


struct ldv_dvb_frontend {
 void *tuner_priv;
};

struct ldv_m88ts2022_config {
 struct ldv_dvb_frontend *fe;
};

struct ldv_i2c_msg;

struct ldv_i2c_adapter {
 int (*master_xfer)(struct ldv_i2c_adapter *adap, struct ldv_i2c_msg *msgs, int num);
};

struct ldv_i2c_client {
 struct ldv_device dev;
 struct ldv_i2c_adapter *adapter;
 void *addr;
};


static inline void *ldv_i2c_get_clientdata(const struct ldv_i2c_client *dev)
{
 return ldv_dev_get_drvdata(&dev->dev);
}

static inline void ldv_i2c_set_clientdata(struct ldv_i2c_client *dev, void *data)
{
 ldv_dev_set_drvdata(&dev->dev, data);
}

struct Data11 {
 int a,b,c;
};




struct ldv_m88ts2022_priv {
 struct ldv_m88ts2022_config cfg;
 struct ldv_i2c_client *client;
};

struct ldv_i2c_msg {
 void *addr;
 int flags;
 int len;
 char *buf;
};
int master_xfer(struct ldv_i2c_adapter *adap, struct ldv_i2c_msg *i2c_msg, int num) {
 int ret = - -3;
 struct ldv_msg *m;
 int i=0;
 while (i < num) {
  m = ldv_msg_alloc();
  if(!m) goto err;
  ret = ldv_msg_fill(m, i2c_msg[i].buf, i2c_msg[i].len);
  if(ret) goto err_fill;
  ret = ldv_submit_msg(m);
  if(ret) goto err_submit;
  i++;
 }
 return i;
err_submit:
err_fill:
 ldv_msg_free(m);
err:
 return ret;
}

int ldv_i2c_transfer(struct ldv_i2c_adapter *adap, struct ldv_i2c_msg *msgs, int num) {



 return adap->master_xfer(adap, msgs, num);

}

static int ldv_m88ts2022_rd_reg(struct ldv_m88ts2022_priv *priv, char reg, char *val) {
 int ret;
 char buf[1];
 struct ldv_i2c_msg msg[2] = {
 {
  .addr = priv->client->addr,
  .flags = 0,
  .len = 1,
  .buf = &reg,
 }, {
  .addr = priv->client->addr,
  .flags = 1,
  .len = 1,
  .buf = buf,
 }
 };
 ret = ldv_i2c_transfer(priv->client->adapter, msg, 2);
 if (ret == 2) {
  memcpy(val, buf, 1);
  ret = 0;
 } else {
  ret = -1;
 }
 return ret;
}

int alloc_12(struct ldv_i2c_client *client) {
 unsigned char chip_id;
 int ret;
 struct ldv_m88ts2022_config *cfg = (struct ldv_m88ts2022_config *)client->dev.platform_data;
 struct ldv_dvb_frontend *fe = cfg->fe;
 struct ldv_m88ts2022_priv *priv = (struct ldv_m88ts2022_priv*)ldv_malloc(sizeof(struct ldv_m88ts2022_priv));
 if(!priv) { ret=- -3; goto err;}
 memcpy(&priv->cfg, cfg, sizeof(struct ldv_m88ts2022_config));
 priv->client = client;

 ret = ldv_m88ts2022_rd_reg(priv, 0x00, &chip_id);
 if(ret) goto err;

 switch (chip_id) {
  case 0xc3:
  case 0x83:
  break;
  default:
  goto err;
 };

 fe->tuner_priv = priv;
 ldv_i2c_set_clientdata(client, priv);
 return 0;
err:
 free(priv);

 return ret;
}

void free_12(struct ldv_i2c_client *client) {
 struct ldv_m88ts2022_config *cfg = (struct ldv_m88ts2022_config *)client->dev.platform_data;
 struct ldv_dvb_frontend *fe = cfg->fe;
 fe->tuner_priv = 0;
 void *priv = (struct ldv_m88ts2022_priv *)ldv_i2c_get_clientdata(client);
 if(priv) {
  free(priv);
 }
}

void entry_point(void) {
 struct ldv_i2c_client *client = (struct ldv_i2c_client *)ldv_malloc(sizeof(struct ldv_i2c_client));
 if(!client) goto err;
 struct ldv_m88ts2022_config *cfg = (struct ldv_m88ts2022_config *)
   ldv_malloc(sizeof(struct ldv_m88ts2022_config));
 if(!cfg) { goto err_cfg; }
 client->dev.platform_data = cfg;
 struct ldv_dvb_frontend *fe = (struct ldv_dvb_frontend *)
   ldv_malloc(sizeof(struct ldv_dvb_frontend));
 if(!fe) { goto err_fe; }
 cfg->fe = fe;

 void *addr = (void *)ldv_malloc(sizeof(int));
 if(!addr) { goto err_addr; }
 client->addr = addr;

 struct ldv_i2c_adapter *adapter = (struct ldv_i2c_adapter *)
   ldv_malloc(sizeof(struct ldv_i2c_adapter));
 if(!adapter) { goto err_adapter; }
 client->adapter = adapter;

 adapter->master_xfer = master_xfer;

 if(alloc_12(client)==0) {
  free_12(client);
 }

 free(adapter);
err_adapter:
 free(addr);
err_addr:
 free(fe);
err_fe:
 free(cfg);
err_cfg:
 free(client);
err:
 ldv_destroy_msgs();
 return;
}

void main(void) {
     entry_point();
}
