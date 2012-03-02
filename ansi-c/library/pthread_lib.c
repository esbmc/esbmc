#include <stdbool.h>
#include <pthread.h>

#include "intrinsics.h"

struct __pthread_start_data {
  __ESBMC_thread_start_func_type func;
  void *start_arg;
};

struct __pthread_start_data __ESBMC_get_thread_internal_data(unsigned int tid);
void __ESBMC_set_thread_internal_data(unsigned int tid,
                                      struct __pthread_start_data data);

#define __ESBMC_mutex_lock_field(a) ((a).__data.__lock)
#define __ESBMC_mutex_count_field(a) ((a).__data.__count)
#define __ESBMC_mutex_owner_field(a) ((a).__data.__owner)
#define __ESBMC_cond_lock_field(a) ((a).__data.__lock)
#define __ESBMC_cond_futex_field(a) ((a).__data.__futex)
#define __ESBMC_cond_nwaiters_field(a) ((a).__data.__nwaiters)
#define __ESBMC_cond_broadcast_seq_field(a) ((a).__data.__broadcast_seq)
#define __ESBMC_rwlock_field(a) ((a).__data.__lock)

/* Global tracking data. Should all initialize to 0 / false */
static _Bool pthread_thread_running[__ESBMC_constant_infinity_uint];
static _Bool pthread_thread_ended[__ESBMC_constant_infinity_uint];
static void *pthread_end_values[__ESBMC_constant_infinity_uint];

static unsigned int num_total_threads = 0;
static unsigned int num_threads_running = 0;
static unsigned int count_wait = 0;
static unsigned int count_lock = 0;

void
pthread_start_main_hook(void)
{
__ESBMC_ATOMIC:
  num_total_threads++;
__ESBMC_ATOMIC:
  num_threads_running++;
}

void
pthread_end_main_hook(void)
{
__ESBMC_ATOMIC:
  num_threads_running--;
}

void
pthread_trampoline(void)
{
__ESBMC_hide:
  struct __pthread_start_data startdata;
  unsigned int threadid;
  void *exit_val;

  threadid = __ESBMC_get_thread_id();
  startdata = __ESBMC_get_thread_internal_data(threadid);

  // Don't cause a context switch when dereferencing this pointer, which hits
  // global pointers (i.e., the function pointer). This is an optimisation.
__ESBMC_ATOMIC:
  exit_val = startdata.func(startdata.start_arg);

  __ESBMC_atomic_begin();
  threadid = __ESBMC_get_thread_id();
  pthread_end_values[threadid] = exit_val;
  pthread_thread_ended[threadid] = true;
  num_threads_running--;
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end(); // Never reached; doesn't matter.
  return;
}

int
pthread_create(pthread_t *thread, const pthread_attr_t *attr,
               void *(*start_routine) (void *), void *arg)
{
__ESBMC_hide:
  unsigned int thread_id;
  struct __pthread_start_data startdata = { start_routine, arg };

  __ESBMC_atomic_begin();
  thread_id = __ESBMC_spawn_thread(pthread_trampoline);
  num_total_threads++;
  num_threads_running++;
  pthread_thread_running[thread_id] = true;
  __ESBMC_set_thread_internal_data(thread_id, startdata);

  // pthread_t is actually an unsigned long int; identify a thread using just
  // its thread number.
  *thread = thread_id;

  __ESBMC_atomic_end();
}

void
pthread_exit(void *retval)
{
__ESBMC_hide:
  __ESBMC_atomic_begin();
  unsigned int threadid = __ESBMC_get_thread_id();
  pthread_end_values[threadid] = retval;
  pthread_thread_ended[threadid] = true;
  num_threads_running--;
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end();
}

pthread_t
pthread_self(void)
{

  return __ESBMC_get_thread_id();
}

int
pthread_join(pthread_t thread, void **retval)
{
__ESBMC_hide:
  __ESBMC_atomic_begin();

  // Assume that it's no longer running. This is dodgy because we're fetching
  // explicit state from inside the model checker, but fine because we're not
  // trying to inject any nondeterminism anywhere.
  __ESBMC_assume(pthread_thread_ended[thread]);

  // Fetch exit code

  if (retval != NULL)
    *retval = pthread_end_values[thread];

  __ESBMC_atomic_end();

  return 0;
}

int pthread_mutex_init(
  pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr)
{
  __ESBMC_HIDE:
  __ESBMC_mutex_lock_field(*mutex)=0;
  __ESBMC_mutex_count_field(*mutex)=0;
  __ESBMC_mutex_owner_field(*mutex)=0;
  return 0;
}

int pthread_mutex_lock(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:
  __ESBMC_yield();
  __ESBMC_assume(!__ESBMC_mutex_lock_field(*mutex));
  __ESBMC_atomic_begin();
  __ESBMC_mutex_lock_field(*mutex)=1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_lock_check(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:
  _Bool unlocked = 1;
  _Bool deadlock_mutex=0;

  __ESBMC_yield();
  __ESBMC_atomic_begin();
  unlocked = (__ESBMC_mutex_lock_field(*mutex)==0);

  if (unlocked)
    __ESBMC_mutex_lock_field(*mutex)=1;
  else
    count_lock++;
  __ESBMC_atomic_end();

  if (__ESBMC_mutex_lock_field(*mutex)==0)
	count_lock--;

  if (!unlocked)
  {
	deadlock_mutex = (count_lock == num_threads_running);
	__ESBMC_assert(!deadlock_mutex,"deadlock detected with mutex lock");
    __ESBMC_assume(deadlock_mutex);
  }

  return 0;
}

int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
  return 0; // we never fail
}

static void do_pthread_mutex_unlock(pthread_mutex_t *mutex, _Bool assrt)
{
  __ESBMC_HIDE:
  __ESBMC_atomic_begin();
  if (assrt)
    __ESBMC_assert(__ESBMC_mutex_lock_field(*mutex), "must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex)=0;
  __ESBMC_atomic_end();
  return;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
  do_pthread_mutex_unlock(mutex, 0);
  return 0;
}

int pthread_mutex_unlock_check(pthread_mutex_t *mutex)
{
  do_pthread_mutex_unlock(mutex, 1);
  return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex)
{ }

int pthread_rwlock_destroy(pthread_rwlock_t *lock)
{ }

int pthread_rwlock_init(pthread_rwlock_t *lock,
  const pthread_rwlockattr_t *attr)
{ __ESBMC_HIDE: __ESBMC_rwlock_field(*lock)=0; }

int pthread_rwlock_rdlock(pthread_rwlock_t *lock)
{ /* TODO */ }

int pthread_rwlock_tryrdlock(pthread_rwlock_t *lock)
{ /* TODO */ }

int pthread_rwlock_trywrlock(pthread_rwlock_t *lock)
{ __ESBMC_HIDE:
  __ESBMC_atomic_begin();
  if(__ESBMC_rwlock_field(*lock)) { __ESBMC_atomic_end(); return 1; }
  __ESBMC_rwlock_field(*lock)=1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_unlock(pthread_rwlock_t *lock)
{ __ESBMC_HIDE: __ESBMC_rwlock_field(*lock)=0; }

int pthread_rwlock_wrlock(pthread_rwlock_t *lock)
{ __ESBMC_HIDE:
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_rwlock_field(*lock));
  __ESBMC_rwlock_field(*lock)=1;
  __ESBMC_atomic_end();
  return 0; // we never fail
}

#if 0
/* FUNCTION: pthread_cond_broadcast */

#ifndef __ESBMC_PTHREAD_H_INCLUDED
#include <pthread.h>
#define __ESBMC_PTHREAD_H_INCLUDED
#endif

int pthread_cond_broadcast(pthread_cond_t *cond)
{
  //__ESBMC_HIDE:
  //printf("broadcast_counter: %d", __ESBMC_cond_broadcast_seq_field(*cond));
  //__ESBMC_cond_broadcast_seq_field(*cond)=1;
  //printf("broadcast_counter: %d", __ESBMC_cond_broadcast_seq_field(*cond));
  __ESBMC_cond_broadcast_seq_field(*cond)=1;
  __ESBMC_assert(__ESBMC_cond_broadcast_seq_field(*cond),"__ESBMC_cond_broadcast_seq_field(*cond)");
  return 0; // we never fail
}
#endif

int pthread_cond_init(
  pthread_cond_t *cond,
  __const pthread_condattr_t *cond_attr)
{
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*cond)=0;
  __ESBMC_cond_broadcast_seq_field(*cond)=0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_cond_destroy(pthread_cond_t *__cond)
{
  __ESBMC_HIDE:
  __ESBMC_cond_lock_field(*__cond)=0;
  return 0;
}

extern int pthread_cond_signal(pthread_cond_t *__cond)
{
  __ESBMC_HIDE:
  __ESBMC_cond_lock_field(*__cond)=0;

  return 0;
}

static void do_pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex,
			_Bool assrt)
{
  __ESBMC_HIDE:
  _Bool deadlock_wait=0;

  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*cond)=1;
  if (assrt)
    __ESBMC_assert(__ESBMC_mutex_lock_field(*mutex),"pthread_cond_wait must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex)=0;
  ++count_wait;
  //__ESBMC_atomic_end();

  //__ESBMC_atomic_begin();
  if (assrt) {
    deadlock_wait = (count_wait == num_threads_running);
    __ESBMC_assert(!deadlock_wait,"deadlock detected with pthread_cond_wait");
  }
  __ESBMC_assume(/*deadlock_wait ||*/ __ESBMC_cond_lock_field(*cond)==0);
  --count_wait;
  __ESBMC_atomic_end();
  __ESBMC_mutex_lock_field(*mutex)=1;

  return;
}

int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{

  do_pthread_cond_wait(cond, mutex, 0);
  return 0;
}

int pthread_cond_wait_check(pthread_cond_t *cond, pthread_mutex_t *mutex)
{

  do_pthread_cond_wait(cond, mutex, 1);
  return 0;
}
