/* ESBMC operational model for ISO C11 <threads.h> (7.26).
 *
 * Based on glibc's <threads.h> but heavily simplified: this header is not
 * meant to be verified itself, only used as a model so that user code which
 * #includes <threads.h> can be parsed and analysed.
 *
 * The C11 thread API is implemented on top of ESBMC's existing pthread
 * operational model, mirroring how glibc layers C11 threads over pthreads.
 */

#ifndef _THREADS_H
#define _THREADS_H 1

#include <time.h>
#include <pthread.h>

#ifndef thread_local
#define thread_local _Thread_local
#endif

#define TSS_DTOR_ITERATIONS 4

/* C11 thread types map directly onto ESBMC's pthread types. */
typedef pthread_t       thrd_t;
typedef pthread_key_t   tss_t;
typedef pthread_mutex_t mtx_t;
typedef pthread_cond_t  cnd_t;
typedef int             once_flag;

#define ONCE_FLAG_INIT 0

typedef int  (*thrd_start_t)(void *);
typedef void (*tss_dtor_t)(void *);

/* Exit and error codes.  */
enum
{
  thrd_success  = 0,
  thrd_busy     = 1,
  thrd_error    = 2,
  thrd_nomem    = 3,
  thrd_timedout = 4
};

/* Mutex types.  */
enum
{
  mtx_plain     = 0,
  mtx_recursive = 1,
  mtx_timed     = 2
};

/* Threads functions.  */
extern int     thrd_create(thrd_t *__thr, thrd_start_t __func, void *__arg);
extern int     thrd_equal(thrd_t __lhs, thrd_t __rhs);
extern thrd_t  thrd_current(void);
extern int     thrd_sleep(const struct timespec *__time_point,
                          struct timespec *__remaining);
extern void    thrd_exit(int __res) __attribute__((__noreturn__));
extern int     thrd_detach(thrd_t __thr);
extern int     thrd_join(thrd_t __thr, int *__res);
extern void    thrd_yield(void);

/* Mutex functions.  */
extern int  mtx_init(mtx_t *__mutex, int __type);
extern int  mtx_lock(mtx_t *__mutex);
extern int  mtx_timedlock(mtx_t *__restrict __mutex,
                          const struct timespec *__restrict __time_point);
extern int  mtx_trylock(mtx_t *__mutex);
extern int  mtx_unlock(mtx_t *__mutex);
extern void mtx_destroy(mtx_t *__mutex);

/* Call a function exactly once.  */
extern void call_once(once_flag *__flag, void (*__func)(void));

/* Condition variable functions.  */
extern int  cnd_init(cnd_t *__cond);
extern int  cnd_signal(cnd_t *__cond);
extern int  cnd_broadcast(cnd_t *__cond);
extern int  cnd_wait(cnd_t *__cond, mtx_t *__mutex);
extern int  cnd_timedwait(cnd_t *__restrict __cond,
                          mtx_t *__restrict __mutex,
                          const struct timespec *__restrict __time_point);
extern void cnd_destroy(cnd_t *__cond);

/* Thread-specific storage functions.  */
extern int   tss_create(tss_t *__tss_id, tss_dtor_t __destructor);
extern void *tss_get(tss_t __tss_id);
extern int   tss_set(tss_t __tss_id, void *__val);
extern void  tss_delete(tss_t __tss_id);

#endif /* _THREADS_H */
