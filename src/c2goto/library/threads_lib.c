/* ESBMC operational model for ISO C11 <threads.h>.
 *
 * The C11 thread API is implemented as thin wrappers around ESBMC's
 * pthread operational model, mirroring how glibc layers C11 threads
 * over pthreads. Translation of return codes follows glibc:
 * any non-zero pthread return becomes thrd_error.
 */

#include <threads.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

/* Internal pthread plumbing reused by the C11 trampoline. */
typedef void *(*__ESBMC_thread_start_func_type)(void *);

struct __pthread_start_data
{
  __ESBMC_thread_start_func_type func;
  void *start_arg;
};

unsigned int __ESBMC_spawn_thread(void (*)(void));
struct __pthread_start_data __ESBMC_get_thread_internal_data(pthread_t tid);
void __ESBMC_set_thread_internal_data(
  pthread_t tid,
  struct __pthread_start_data data);
void __ESBMC_terminate_thread(void);
pthread_t __ESBMC_get_thread_id(void);

extern unsigned short int __ESBMC_num_total_threads;
extern unsigned short int __ESBMC_num_threads_running;
extern unsigned short int __ESBMC_blocked_threads_count;

__attribute__((annotate("__ESBMC_inf_size")))
extern _Bool __ESBMC_pthread_thread_running[1];
__attribute__((annotate("__ESBMC_inf_size")))
extern _Bool __ESBMC_pthread_thread_ended[1];
__attribute__((annotate("__ESBMC_inf_size")))
extern void *__ESBMC_pthread_end_values[1];

/* Per-thread storage of the C11 start routine.  Because pthread expects
 * void *(*)(void *) but C11 provides int (*)(void *), we cannot reuse
 * the pthread trampoline directly. */
__attribute__((annotate("__ESBMC_inf_size")))
static thrd_start_t __ESBMC_c11_thrd_func[1];
__attribute__((annotate("__ESBMC_inf_size")))
static void *__ESBMC_c11_thrd_arg[1];

static void __ESBMC_c11_thrd_trampoline(void)
{
__ESBMC_HIDE:;
  pthread_t tid = __ESBMC_get_thread_id();
  thrd_start_t f = __ESBMC_c11_thrd_func[tid];
  void *a = __ESBMC_c11_thrd_arg[tid];

  int rc = f(a);

  __ESBMC_atomic_begin();
  tid = __ESBMC_get_thread_id();
  __ESBMC_pthread_end_values[tid] = (void *)(intptr_t)rc;
  __ESBMC_pthread_thread_ended[tid] = 1;
  __ESBMC_num_threads_running--;
  __ESBMC_assume(__ESBMC_blocked_threads_count == 0);
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end();

  __ESBMC_assume(0);
}

int thrd_create(thrd_t *thr, thrd_start_t func, void *arg)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t tid = __ESBMC_spawn_thread(__ESBMC_c11_thrd_trampoline);
  __ESBMC_num_total_threads++;
  __ESBMC_num_threads_running++;
  __ESBMC_pthread_thread_running[tid] = 1;
  __ESBMC_pthread_thread_ended[tid] = 0;
  __ESBMC_pthread_end_values[tid] = NULL;
  __ESBMC_c11_thrd_func[tid] = func;
  __ESBMC_c11_thrd_arg[tid] = arg;
  *thr = tid;
  __ESBMC_atomic_end();
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
  (void)time_point;
  if (remaining)
  {
    remaining->tv_sec = 0;
    remaining->tv_nsec = 0;
  }
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
  int rc = pthread_join(thr, &retval);
  if (rc != 0)
    return thrd_error;
  if (res)
    *res = (int)(intptr_t)retval;
  return thrd_success;
}

void thrd_yield(void)
{
__ESBMC_HIDE:;
  /* Yielding is a hint; ESBMC explores all interleavings already. */
}

/* Mutex functions.  C11 mutex types (mtx_plain, mtx_recursive, mtx_timed)
 * are accepted but treated uniformly by ESBMC's pthread mutex model. */
int mtx_init(mtx_t *m, int type)
{
__ESBMC_HIDE:;
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

/* call_once: a one-shot guarded call.  Atomic check-and-set is enough
 * for verification; pthread_once is not provided by ESBMC's model. */
void call_once(once_flag *flag, void (*func)(void))
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  if (*flag == 0)
  {
    *flag = 1;
    __ESBMC_atomic_end();
    func();
    return;
  }
  __ESBMC_atomic_end();
}

/* Condition variables.  */
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
  (void)time_point;
  return pthread_cond_wait(c, m) == 0 ? thrd_success : thrd_error;
}

void cnd_destroy(cnd_t *c)
{
__ESBMC_HIDE:;
  pthread_cond_destroy(c);
}

/* Thread-specific storage.  */
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
  /* ESBMC does not implement pthread_key_delete; key teardown is a no-op. */
  (void)key;
}
