/* ESBMC operational model for ISO C11 <threads.h>: thin wrappers over the
 * pthread OM, mirroring how glibc layers C11 threads on pthreads. */

#include <threads.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

/* Bridge from C11's int(*)(void*) to pthread's void*(*)(void*). A private
 * heap-allocated bundle per spawn beats a shared inf_size array: it gives
 * each thread its own start-data object so the verifier doesn't track
 * cross-thread shared state. */
struct __c11_thrd_thunk_data
{
  thrd_start_t func;
  void *arg;
};

static void *__c11_thrd_thunk(void *p)
{
__ESBMC_HIDE:;
  struct __c11_thrd_thunk_data t = *(struct __c11_thrd_thunk_data *)p;
  free(p);
  return (void *)(intptr_t)t.func(t.arg);
}

int thrd_create(thrd_t *thr, thrd_start_t func, void *arg)
{
__ESBMC_HIDE:;
  struct __c11_thrd_thunk_data *t = malloc(sizeof *t);
  if (!t)
    return thrd_nomem;
  t->func = func;
  t->arg = arg;
  pthread_create(thr, NULL, __c11_thrd_thunk, t);
  return thrd_success;
}

int thrd_equal(thrd_t a, thrd_t b)
{
__ESBMC_HIDE:;
  return a == b;
}

thrd_t thrd_current(void)
{
__ESBMC_HIDE:;
  return pthread_self();
}

int thrd_sleep(const struct timespec *time_point, struct timespec *remaining)
{
__ESBMC_HIDE:;
  /* OM: real-time sleep is unobservable; ESBMC explores all schedules. On
   * success the spec leaves *remaining untouched, so do nothing here. */
  (void)time_point;
  (void)remaining;
  return 0;
}

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-noreturn"
void thrd_exit(int res)
{
__ESBMC_HIDE:;
  pthread_exit((void *)(intptr_t)res);
}
#pragma clang diagnostic pop

int thrd_detach(thrd_t thr)
{
__ESBMC_HIDE:;
  return pthread_detach(thr) == 0 ? thrd_success : thrd_error;
}

int thrd_join(thrd_t thr, int *res)
{
__ESBMC_HIDE:;
  void *retval = NULL;
  if (pthread_join(thr, &retval) != 0)
    return thrd_error;
  if (res)
    *res = (int)(intptr_t)retval;
  return thrd_success;
}

void thrd_yield(void)
{
__ESBMC_HIDE:;
  /* ESBMC explores all interleavings; yield is a no-op. */
}

int mtx_init(mtx_t *m, int type)
{
__ESBMC_HIDE:;
  /* OM: mtx_recursive / mtx_timed not distinguished. */
  (void)type;
  return pthread_mutex_init(m, NULL) == 0 ? thrd_success : thrd_error;
}

int mtx_lock(mtx_t *m)
{
__ESBMC_HIDE:;
  return pthread_mutex_lock(m) == 0 ? thrd_success : thrd_error;
}

int mtx_timedlock(mtx_t *m, const struct timespec *time_point)
{
__ESBMC_HIDE:;
  /* OM: timeout dropped, models the blocking branch only — sound
   * over-approximation, never returns thrd_timedout. */
  (void)time_point;
  return pthread_mutex_lock(m) == 0 ? thrd_success : thrd_error;
}

int mtx_trylock(mtx_t *m)
{
__ESBMC_HIDE:;
  return pthread_mutex_trylock(m) == 0 ? thrd_success : thrd_busy;
}

int mtx_unlock(mtx_t *m)
{
__ESBMC_HIDE:;
  return pthread_mutex_unlock(m) == 0 ? thrd_success : thrd_error;
}

void mtx_destroy(mtx_t *m)
{
__ESBMC_HIDE:;
  pthread_mutex_destroy(m);
}

void call_once(once_flag *flag, void (*func)(void))
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  if (flag->__data == 0)
  {
    flag->__data = 1;
    __ESBMC_atomic_end();
    func();
    return;
  }
  __ESBMC_atomic_end();
}

int cnd_init(cnd_t *c)
{
__ESBMC_HIDE:;
  return pthread_cond_init(c, NULL) == 0 ? thrd_success : thrd_error;
}

int cnd_signal(cnd_t *c)
{
__ESBMC_HIDE:;
  return pthread_cond_signal(c) == 0 ? thrd_success : thrd_error;
}

int cnd_broadcast(cnd_t *c)
{
__ESBMC_HIDE:;
  return pthread_cond_broadcast(c) == 0 ? thrd_success : thrd_error;
}

int cnd_wait(cnd_t *c, mtx_t *m)
{
__ESBMC_HIDE:;
  return pthread_cond_wait(c, m) == 0 ? thrd_success : thrd_error;
}

int cnd_timedwait(cnd_t *c, mtx_t *m, const struct timespec *time_point)
{
__ESBMC_HIDE:;
  /* OM: timeout dropped, see mtx_timedlock. */
  (void)time_point;
  return pthread_cond_wait(c, m) == 0 ? thrd_success : thrd_error;
}

void cnd_destroy(cnd_t *c)
{
__ESBMC_HIDE:;
  pthread_cond_destroy(c);
}

int tss_create(tss_t *key, tss_dtor_t destructor)
{
__ESBMC_HIDE:;
  return pthread_key_create(key, destructor) == 0 ? thrd_success : thrd_error;
}

void *tss_get(tss_t key)
{
__ESBMC_HIDE:;
  return pthread_getspecific(key);
}

int tss_set(tss_t key, void *val)
{
__ESBMC_HIDE:;
  return pthread_setspecific(key, val) == 0 ? thrd_success : thrd_error;
}

void tss_delete(tss_t key)
{
__ESBMC_HIDE:;
  /* ESBMC's pthread OM has no pthread_key_delete; teardown is a no-op. */
  (void)key;
}
