#include <errno.h>

#include "../headers/pthreadtypes.hs"
#include "intrinsics.h"

typedef void *(*__ESBMC_thread_start_func_type)(void *);
void __ESBMC_terminate_thread(void);
unsigned int __ESBMC_spawn_thread(void (*)(void));

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
__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_thread_running[1];

__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_thread_ended[1];

__attribute__((used))
__attribute__((annotate("__ESBMC_inf_size")))
void *__ESBMC_pthread_end_values[1];

unsigned int __ESBMC_num_total_threads = 0;
unsigned int __ESBMC_num_threads_running = 0;
unsigned int __ESBMC_blocked_threads_count = 0;

pthread_t __ESBMC_get_thread_id(void);

void __ESBMC_really_atomic_begin(void);
void __ESBMC_really_atomic_end(void);

/************************** Thread creation and exit **************************/

void
pthread_start_main_hook(void)
{
  __ESBMC_atomic_begin();
  __ESBMC_num_total_threads++;
  __ESBMC_num_threads_running++;
  __ESBMC_atomic_end();
}

void
pthread_end_main_hook(void)
{
  // We want to be able to access this internal accounting data atomically,
  // but that'll never be permitted by POR, which will see the access and try
  // to generate context switches as a result. So, end the main thread in an
  // atomic state, which will prevent everything but the final from-main switch.
  __ESBMC_atomic_begin();
  __ESBMC_num_threads_running--;
}

void
pthread_trampoline(void)
{
  __ESBMC_HIDE:;
  pthread_t threadid = __ESBMC_get_thread_id();
  struct __pthread_start_data startdata =
    __ESBMC_get_thread_internal_data(threadid);

  void *exit_val = startdata.func(startdata.start_arg);

  __ESBMC_atomic_begin();
  threadid = __ESBMC_get_thread_id();
  __ESBMC_pthread_end_values[(int)threadid] = exit_val;
  __ESBMC_pthread_thread_ended[(int)threadid] = 1;
  __ESBMC_num_threads_running--;
  // A thread terminating during a search for a deadlock means there's no
  // deadlock or it can be found down a different path. Proof left as exercise
  // to the reader.
  __ESBMC_assume(__ESBMC_blocked_threads_count == 0);
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end(); // Never reached; doesn't matter.
  return;
}

int
pthread_create(pthread_t *thread, const pthread_attr_t *attr,
  void *(*start_routine)(void *),
  void *arg)
{
  __ESBMC_HIDE:;
  struct __pthread_start_data startdata = { start_routine, arg };

  __ESBMC_atomic_begin();
  pthread_t threadid = __ESBMC_spawn_thread(pthread_trampoline);
  __ESBMC_num_total_threads++;
  __ESBMC_num_threads_running++;
  __ESBMC_pthread_thread_running[threadid] = 1;
  __ESBMC_pthread_thread_ended[threadid] = 0;
  __ESBMC_pthread_end_values[threadid] = NULL;
  __ESBMC_set_thread_internal_data(threadid, startdata);

  // pthread_t is actually an unsigned long int; identify a thread using just
  // its thread number.
  *thread = threadid;

  __ESBMC_atomic_end();
  return 0; // We never fail
}

void
pthread_exit(void *retval)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t threadid = __ESBMC_get_thread_id();
  __ESBMC_pthread_end_values[(int)threadid] = retval;
  __ESBMC_pthread_thread_ended[(int)threadid] = 1;
  __ESBMC_num_threads_running--;
  // A thread terminating during a search for a deadlock means there's no
  // deadlock or it can be found down a different path. Proof left as exercise
  // to the reader.
  __ESBMC_assume(__ESBMC_blocked_threads_count == 0);
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end();
}

pthread_t
pthread_self(void)
{
  return __ESBMC_get_thread_id();
}

int
pthread_join_switch(pthread_t thread, void **retval)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  // Detect whether the target thread has ended or not. If it isn't, mark us as
  // waiting for its completion. That fact can be used for deadlock detection
  // elsewhere.
  _Bool ended = __ESBMC_pthread_thread_ended[(int)thread];
  if (!ended) {
    __ESBMC_blocked_threads_count++;
    // If there are now no more threads unblocked, croak.
    __ESBMC_assert(__ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
                   "Deadlocked state in pthread_join");
  }

  // Fetch exit code
  if (retval != NULL)
    *retval = __ESBMC_pthread_end_values[(int)thread];

  // In all circumstances, allow a switch away from this thread to permit
  // deadlock checking,
  __ESBMC_atomic_end();

  // But if this thread is blocked, don't allow for any further execution.
  __ESBMC_assume(ended);

  return 0;
}

int
pthread_join_noswitch(pthread_t thread, void **retval)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  // If the other thread hasn't ended, assume false, because further progress
  // isn't going to be made. Wait for an interleaving where this is true
  // instead. This function isn't designed for deadlock detection.
  _Bool ended = __ESBMC_pthread_thread_ended[(int)thread];
  __ESBMC_assume(ended);

  // Fetch exit code
  if (retval != NULL)
    *retval = __ESBMC_pthread_end_values[(int)thread];

  __ESBMC_really_atomic_end();

  return 0;
}

/************************* Mutex manipulation routines ************************/

int
pthread_mutex_init(
  pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr)
{
  __ESBMC_HIDE:;
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_mutex_count_field(*mutex) = 0;
  __ESBMC_mutex_owner_field(*mutex) = 0;
  return 0;
}

int
pthread_mutex_lock_noassert(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_mutex_lock_field(*mutex));
  __ESBMC_mutex_lock_field(*mutex) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_mutex_lock_nocheck(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_mutex_lock_field(*mutex));
  __ESBMC_mutex_lock_field(*mutex) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_mutex_unlock_noassert(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  __ESBMC_mutex_lock_field(*mutex) = 0;
  return 0;
}

int
pthread_mutex_unlock_nocheck(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(__ESBMC_mutex_lock_field(*mutex), "must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_mutex_lock_check(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  _Bool unlocked = 1;

  __ESBMC_atomic_begin();
  unlocked = (__ESBMC_mutex_lock_field(*mutex) == 0);

  if (unlocked) {
    __ESBMC_mutex_lock_field(*mutex) = 1;
  } else {
    // Deadlock foo
    __ESBMC_blocked_threads_count++;
    // No more threads to run -> croak.
    __ESBMC_assert(__ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
                   "Deadlocked state in pthread_mutex_lock");
  }

  // Switch away for deadlock detection and so forth...
  __ESBMC_atomic_end();

  // ... but don't allow execution further if it was locked.
  __ESBMC_assume(unlocked);

  return 0;
}

int
pthread_mutex_unlock_check(pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(__ESBMC_mutex_lock_field(*mutex), "must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_mutex_trylock(pthread_mutex_t *mutex)
{
  if (__ESBMC_mutex_lock_field(*mutex) != 0) {
    return EBUSY;
  } else {
    pthread_mutex_lock(mutex);
    return 0;
  }
}

int
pthread_mutex_destroy(pthread_mutex_t *mutex)
{
  return 0;
}

int
pthread_rwlock_destroy(pthread_rwlock_t *lock)
{
  return 0;
}

/************************ rwlock mainpulation routines ************************/

int
pthread_rwlock_init(pthread_rwlock_t *lock, const pthread_rwlockattr_t *attr)
{
  __ESBMC_HIDE:;
  __ESBMC_rwlock_field(*lock) = 0;
  return 0;
}

int
pthread_rwlock_rdlock(pthread_rwlock_t *lock)
{
  return 0;
}

int
pthread_rwlock_tryrdlock(pthread_rwlock_t *lock)
{
  return 0;
}

int
pthread_rwlock_trywrlock(pthread_rwlock_t *lock)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  if (__ESBMC_rwlock_field(*lock)) {
    __ESBMC_atomic_end();
    return 1;
  }
  __ESBMC_rwlock_field(*lock) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_rwlock_unlock(pthread_rwlock_t *lock)
{
  __ESBMC_HIDE:;
  __ESBMC_rwlock_field(*lock) = 0;
  return 0;
}

int
pthread_rwlock_wrlock(pthread_rwlock_t *lock)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_rwlock_field(*lock));
  __ESBMC_rwlock_field(*lock) = 1;
  __ESBMC_atomic_end();
  return 0; // we never fail
}

/************************ condvar mainpulation routines ***********************/

// this is currently unimplemented.
int pthread_cond_broadcast(pthread_cond_t *cond)
{
  __ESBMC_HIDE:;
  return 0;
}

int
pthread_cond_init(
  pthread_cond_t *cond, __const pthread_condattr_t *cond_attr)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*cond) = 0;
  __ESBMC_cond_broadcast_seq_field(*cond) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int
pthread_cond_destroy(pthread_cond_t *__cond)
{
  __ESBMC_HIDE:;
  __ESBMC_cond_lock_field(*__cond) = 0;
  return 0;
}

extern int
pthread_cond_signal(pthread_cond_t *__cond)
{
  __ESBMC_HIDE:;
  __ESBMC_cond_lock_field(*__cond) = 0;
  return 0;
}

static void
do_pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex, _Bool assrt)
{
  __ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  if (assrt)
    __ESBMC_assert(__ESBMC_mutex_lock_field( *mutex),
                   "caller must hold pthread mutex lock in pthread_cond_wait");

  // Unlock mutex; register us as waiting on condvar; context switch
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_cond_lock_field(*cond) = 1;

  // Technically in the gap below, we are blocked. So mark ourselves thus. If
  // all other threads are (or become) blocked, then deadlock occurred, which
  // this helps detect.
  __ESBMC_blocked_threads_count++;
  // No more threads to run -> croak.
  __ESBMC_assert(__ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
                 "Deadlocked state in pthread_mutex_lock");

  __ESBMC_atomic_end();

  // Other thread activity to happen in this gap

  __ESBMC_atomic_begin();

  // Have we been signalled?
  _Bool signalled = __ESBMC_cond_lock_field(*cond) == 0;

  // Don't consider any other interleavings aside from the ones where we've
  // been signalled. As with mutexes, we should discard this trace and look
  // for one where we /have/ been signalled instead. There's no use in
  // switching away from this thread and looking for deadlock; if that's
  // reachable, it'll be found by the context switch earlier in this function.
  __ESBMC_assume(signalled);
  // We're no longer blocked.
  __ESBMC_blocked_threads_count--;

  __ESBMC_atomic_end();

  // You're permitted to signal a condvar while you hold its mutex, so we have
  // to allow a context switch before reaquiring the mutex to handle that
  // situation
  pthread_mutex_lock(mutex);

  return;
}

int
pthread_cond_wait_nocheck(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  do_pthread_cond_wait(cond, mutex, 0);
  return 0;
}

int
pthread_cond_wait_check(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
  __ESBMC_HIDE:;
  do_pthread_cond_wait(cond, mutex, 1);
  return 0;
}
