/* Operational model for C11 <threads.h>, layered on ESBMC's pthread model in
 * the same way glibc layers C11 threads over pthreads. See issue #3449.
 *
 * Modelling notes:
 *  - The timed variants (mtx_timedlock, cnd_timedwait) have no clock to block
 *    against, so they choose nondeterministically between the untimed
 *    behaviour and thrd_timedout. That keeps both outcomes reachable rather
 *    than silently dropping the timeout path.
 *  - thrd_sleep does not advance any clock; it returns 0 (slept fully).
 */

#include <pthread.h>
#include <stddef.h>
#include <threads.h>

int nondet_int(void);
_Bool nondet_bool(void);

int thrd_create(thrd_t *thr, thrd_start_t func, void *arg)
{
__ESBMC_HIDE:;
  /* C11's entry point returns int, pthread's returns void*. The pthread model
   * calls the stored pointer directly and ESBMC resolves the callee from the
   * pointer's value set, so the cast is enough: no trampoline is needed and
   * the int result is widened to the void* exit value that thrd_join reads
   * back. */
  if (pthread_create(thr, NULL, (void *(*)(void *))func, arg) != 0)
    return thrd_error;

  return thrd_success;
}

int thrd_equal(thrd_t a, thrd_t b)
{
__ESBMC_HIDE:;
  return pthread_equal(a, b);
}

thrd_t thrd_current(void)
{
__ESBMC_HIDE:;
  return pthread_self();
}

int thrd_sleep(const struct timespec *duration, struct timespec *remaining)
{
__ESBMC_HIDE:;
  /* No clock is modelled: report the full interval as elapsed. */
  if (remaining != NULL)
  {
    remaining->tv_sec = 0;
    remaining->tv_nsec = 0;
  }
  return 0;
}

void thrd_yield(void)
{
__ESBMC_HIDE:;
  __ESBMC_yield();
}

_Noreturn void thrd_exit(int res)
{
__ESBMC_HIDE:;
  pthread_exit((void *)(unsigned long)(unsigned int)res);
}

int thrd_detach(thrd_t thr)
{
__ESBMC_HIDE:;
  return pthread_detach(thr) == 0 ? thrd_success : thrd_error;
}

int thrd_join(thrd_t thr, int *res)
{
__ESBMC_HIDE:;
  void *exit_value;
  if (pthread_join(thr, &exit_value) != 0)
    return thrd_error;

  if (res != NULL)
    *res = (int)(unsigned int)(unsigned long)exit_value;

  return thrd_success;
}

int mtx_init(mtx_t *mtx, int type)
{
__ESBMC_HIDE:;
  if (type & mtx_recursive)
  {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    int r = pthread_mutex_init(mtx, &attr);
    pthread_mutexattr_destroy(&attr);
    return r == 0 ? thrd_success : thrd_error;
  }

  return pthread_mutex_init(mtx, NULL) == 0 ? thrd_success : thrd_error;
}

int mtx_lock(mtx_t *mtx)
{
__ESBMC_HIDE:;
  return pthread_mutex_lock(mtx) == 0 ? thrd_success : thrd_error;
}

int mtx_timedlock(mtx_t *__restrict mtx, const struct timespec *__restrict ts)
{
__ESBMC_HIDE:;
  if (nondet_bool())
    return thrd_timedout;

  return mtx_lock(mtx);
}

int mtx_trylock(mtx_t *mtx)
{
__ESBMC_HIDE:;
  int r = pthread_mutex_trylock(mtx);
  if (r == 0)
    return thrd_success;

  return thrd_busy;
}

int mtx_unlock(mtx_t *mtx)
{
__ESBMC_HIDE:;
  return pthread_mutex_unlock(mtx) == 0 ? thrd_success : thrd_error;
}

void mtx_destroy(mtx_t *mtx)
{
__ESBMC_HIDE:;
  pthread_mutex_destroy(mtx);
}

void call_once(once_flag *flag, void (*func)(void))
{
__ESBMC_HIDE:;
  /* pthread_once has no operational model, so the flag is tested and set
   * directly. The read-modify-write must be atomic, otherwise two racing
   * callers could both observe the flag clear and run func twice. */
  __ESBMC_atomic_begin();
  _Bool run = (*flag == 0);
  if (run)
    *flag = 1;
  __ESBMC_atomic_end();

  if (run)
    func();
}

int cnd_init(cnd_t *cond)
{
__ESBMC_HIDE:;
  return pthread_cond_init(cond, NULL) == 0 ? thrd_success : thrd_error;
}

int cnd_signal(cnd_t *cond)
{
__ESBMC_HIDE:;
  return pthread_cond_signal(cond) == 0 ? thrd_success : thrd_error;
}

int cnd_broadcast(cnd_t *cond)
{
__ESBMC_HIDE:;
  return pthread_cond_broadcast(cond) == 0 ? thrd_success : thrd_error;
}

int cnd_wait(cnd_t *cond, mtx_t *mtx)
{
__ESBMC_HIDE:;
  return pthread_cond_wait(cond, mtx) == 0 ? thrd_success : thrd_error;
}

int cnd_timedwait(
  cnd_t *__restrict cond,
  mtx_t *__restrict mtx,
  const struct timespec *__restrict ts)
{
__ESBMC_HIDE:;
  if (nondet_bool())
    return thrd_timedout;

  return cnd_wait(cond, mtx);
}

void cnd_destroy(cnd_t *cond)
{
__ESBMC_HIDE:;
  pthread_cond_destroy(cond);
}

int tss_create(tss_t *key, tss_dtor_t dtor)
{
__ESBMC_HIDE:;
  return pthread_key_create(key, dtor) == 0 ? thrd_success : thrd_error;
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
  /* pthread_key_delete has no operational model, so the key is simply left
   * allocated. C11 7.26.6.2 makes any later use of a deleted key undefined, so
   * retaining it cannot mask a defined behaviour; it only means the slot is
   * not recycled by a subsequent tss_create. */
  (void)key;
}
