/* C11 <threads.h>.
 *
 * ESBMC ships its own <pthread.h> operational model and puts its header
 * directory on the -isystem path, which shadows glibc's
 * <bits/thread-shared-types.h> with a forwarder to that model. The system
 * <threads.h> expects the glibc original and its private types (__tss_t,
 * __thrd_t, ...), so including it used to fail with "unknown type name
 * '__tss_t'" on glibc and "file not found" on platforms that ship no
 * <threads.h> at all. Provide the header ourselves, layered on the pthread
 * model exactly as glibc layers C11 threads over pthreads. See issue #3449.
 */

#ifndef __ESBMC_THREADS_H
#define __ESBMC_THREADS_H

#include <pthread.h>
#include <time.h>

#define thread_local _Thread_local

/* C11 7.26.5.1: at least 4 rounds of TSS destructor calls. */
#define TSS_DTOR_ITERATIONS 4

typedef pthread_t thrd_t;
typedef pthread_mutex_t mtx_t;
typedef pthread_cond_t cnd_t;
typedef pthread_key_t tss_t;

typedef int (*thrd_start_t)(void *);
typedef void (*tss_dtor_t)(void *);

typedef int once_flag;
#define ONCE_FLAG_INIT 0

enum
{
  mtx_plain = 0,
  mtx_recursive = 1,
  mtx_timed = 2
};

enum
{
  thrd_success = 0,
  thrd_busy = 1,
  thrd_error = 2,
  thrd_nomem = 3,
  thrd_timedout = 4
};

int thrd_create(thrd_t *, thrd_start_t, void *);
int thrd_equal(thrd_t, thrd_t);
thrd_t thrd_current(void);
int thrd_sleep(const struct timespec *, struct timespec *);
void thrd_yield(void);
_Noreturn void thrd_exit(int);
int thrd_detach(thrd_t);
int thrd_join(thrd_t, int *);

int mtx_init(mtx_t *, int);
int mtx_lock(mtx_t *);
int mtx_timedlock(mtx_t *__restrict, const struct timespec *__restrict);
int mtx_trylock(mtx_t *);
int mtx_unlock(mtx_t *);
void mtx_destroy(mtx_t *);

void call_once(once_flag *, void (*)(void));

int cnd_init(cnd_t *);
int cnd_signal(cnd_t *);
int cnd_broadcast(cnd_t *);
int cnd_wait(cnd_t *, mtx_t *);
int cnd_timedwait(cnd_t *__restrict, mtx_t *__restrict, const struct timespec *__restrict);
void cnd_destroy(cnd_t *);

int tss_create(tss_t *, tss_dtor_t);
void *tss_get(tss_t);
int tss_set(tss_t, void *);
void tss_delete(tss_t);

#endif
