#include <errno.h>
#include <pthread.h>
#include <stddef.h>

#define CLEANUP_STACK_CHUNK_SIZE 1024

typedef void *(*__ESBMC_thread_start_func_type)(void *);
void __ESBMC_terminate_thread(void);
unsigned int __ESBMC_spawn_thread(void (*)(void));

struct __pthread_start_data
{
  __ESBMC_thread_start_func_type func;
  void *start_arg;
};

struct __pthread_start_data __ESBMC_get_thread_internal_data(pthread_t tid);
void __ESBMC_set_thread_internal_data(
  pthread_t tid,
  struct __pthread_start_data data);

#define __ESBMC_mutex_lock_field(a) ((a).__lock)
#define __ESBMC_mutex_count_field(a) ((a).__count)
#define __ESBMC_mutex_owner_field(a) ((a).__owner)
#define __ESBMC_cond_lock_field(a) ((a).__lock)
#define __ESBMC_cond_futex_field(a) ((a).__futex)
#define __ESBMC_cond_nwaiters_field(a) ((a).__nwaiters)
#define __ESBMC_rwlock_readers(a) ((a)->__readers)
#define __ESBMC_rwlock_writer(a) ((a)->__writer)

/* Global tracking data. Should all initialize to 0 / false */
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_thread_running[1];

__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_thread_ended[1];

__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_thread_detach[1];

__attribute__((
  annotate("__ESBMC_inf_size"))) void *__ESBMC_pthread_end_values[1];

void(
  __attribute__((annotate("__ESBMC_inf_size"))) *
  __ESBMC_thread_key_destructors[1])(void *);

static pthread_key_t __ESBMC_next_thread_key = 0;

unsigned short int __ESBMC_num_total_threads = 0;
unsigned short int __ESBMC_num_threads_running = 0;
unsigned short int __ESBMC_blocked_threads_count = 0;

/* Per-thread cancellation state. All default to 0:
 *   cancel_requested = false
 *   cancelstate      = PTHREAD_CANCEL_ENABLE  (0)
 *   canceltype       = PTHREAD_CANCEL_DEFERRED (0) */
__attribute__((annotate("__ESBMC_inf_size")))
_Bool __ESBMC_pthread_cancel_requested[1];

__attribute__((
  annotate("__ESBMC_inf_size"))) int __ESBMC_pthread_cancelstate[1];

__attribute__((annotate("__ESBMC_inf_size"))) int __ESBMC_pthread_canceltype[1];

pthread_t __ESBMC_get_thread_id(void);

void __ESBMC_really_atomic_begin(void);
void __ESBMC_really_atomic_end(void);

static _Bool pthread_cleanup_enabled = 0;

typedef struct
{
  void *arg;
  void *function;
} cleanup_entry_t;

cleanup_entry_t __esbmc_cleanup_stack[1]
  __attribute__((annotate("__ESBMC_inf_size")));
size_t __esbmc_cleanup_level[1] __attribute__((annotate("__ESBMC_inf_size")));

size_t __esbmc_get_cleanup_level(void)
{
  pthread_t tid = __ESBMC_get_thread_id();
  return __esbmc_cleanup_level[tid];
}

void __esbmc_set_cleanup_level(int level)
{
  pthread_t tid = __ESBMC_get_thread_id();
  __esbmc_cleanup_level[tid] = level;
}

/************************** Infinite Array Implementation **************************/

/* These internal functions insert_key_value(), search_key() and delete_key()
 * need to be called in an atomic context. */

typedef struct thread_key
{
  pthread_t thread;
  pthread_key_t key;
  const void *value;
} __ESBMC_thread_key;

__attribute__((annotate(
  "__ESBMC_inf_size"))) static __ESBMC_thread_key __ESBMC_pthread_thread_key[1];

static void insert_key_value(pthread_key_t key, const void *value)
{
__ESBMC_HIDE:;
  pthread_t thread = __ESBMC_get_thread_id();
  __ESBMC_pthread_thread_key[thread].thread = thread;
  __ESBMC_pthread_thread_key[thread].key = key;
  __ESBMC_pthread_thread_key[thread].value = value;
}

static __ESBMC_thread_key *search_key(pthread_key_t key)
{
__ESBMC_HIDE:;
  pthread_t thread = __ESBMC_get_thread_id();
  if (
    __ESBMC_pthread_thread_key[thread].thread == thread &&
    __ESBMC_pthread_thread_key[thread].key == key)
  {
    return &__ESBMC_pthread_thread_key[thread];
  }
  return NULL;
}

static int delete_key(__ESBMC_thread_key *l)
{
__ESBMC_HIDE:;
  pthread_t thread = __ESBMC_get_thread_id();
  if (&__ESBMC_pthread_thread_key[thread] == l)
  {
    __ESBMC_pthread_thread_key[thread].thread = 0; // Mark as empty
    return 0;
  }
  return -1;
}

/************************** Thread creation and exit **************************/

void __ESBMC_pthread_start_main_hook(void)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_num_total_threads++;
  __ESBMC_num_threads_running++;
  __ESBMC_atomic_end();
}

void __ESBMC_pthread_end_main_hook(void)
{
__ESBMC_HIDE:;
  // We want to be able to access this internal accounting data atomically,
  // but that'll never be permitted by POR, which will see the access and try
  // to generate context switches as a result. So, end the main thread in an
  // atomic state, which will prevent everything but the final from-main switch.
  __ESBMC_atomic_begin();
  __ESBMC_num_threads_running--;
}

void pthread_exec_key_destructors(void)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  // At thread exit, if a key value has a non-NULL destructor pointer,
  // and the thread has a non-NULL value associated with that key,
  // the value of the key is set to NULL, and then the function pointed to
  // is called with the previously associated value as its sole argument.
  // The order of destructor calls is unspecified if more than one destructor
  // exists for a thread when it exits.
  // source: https://linux.die.net/man/3/pthread_key_create
  for (unsigned long i = 0; i < __ESBMC_next_thread_key; ++i)
  {
    __ESBMC_thread_key *l = search_key(i);
    if (__ESBMC_thread_key_destructors[i] && l->value)
    {
      __ESBMC_thread_key_destructors[i]((void *)l->value);
      delete_key(l);
    }
  }
  __ESBMC_atomic_end();
}

void pthread_trampoline(void)
{
__ESBMC_HIDE:;
  pthread_t threadid = __ESBMC_get_thread_id();
  struct __pthread_start_data startdata =
    __ESBMC_get_thread_internal_data(threadid);

  void *exit_val = startdata.func(startdata.start_arg);

  __ESBMC_atomic_begin();
  threadid = __ESBMC_get_thread_id();
  __ESBMC_pthread_end_values[threadid] = exit_val;
  __ESBMC_pthread_thread_ended[threadid] = 1;
  __ESBMC_num_threads_running--;
  // A thread terminating during a search for a deadlock means there's no
  // deadlock or it can be found down a different path. Proof left as exercise
  // to the reader.
  __ESBMC_assume(__ESBMC_blocked_threads_count == 0);
  pthread_exec_key_destructors();
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end(); // Never reached; doesn't matter.

  // Ensure that we cut all subsequent execution paths.
  __ESBMC_assume(0);
  return;
}

int pthread_create(
  pthread_t *thread,
  const pthread_attr_t *attr,
  void *(*start_routine)(void *),
  void *arg)
{
__ESBMC_HIDE:;
  struct __pthread_start_data startdata = {start_routine, arg};

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

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-noreturn"
void pthread_exit(void *retval)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  // Run key destructors first
  pthread_exec_key_destructors();

  // Run cleanup handlers
  while (pthread_cleanup_enabled && __esbmc_get_cleanup_level() > 0)
    pthread_cleanup_pop(1); // 1 means execute the handler

  pthread_t threadid = __ESBMC_get_thread_id();
  __ESBMC_pthread_end_values[threadid] = retval;
  __ESBMC_pthread_thread_ended[threadid] = 1;
  __ESBMC_num_threads_running--;

  // A thread terminating during a search for a deadlock means there's no
  // deadlock or it can be found down a different path. Proof left as exercise
  // to the reader.
  __ESBMC_assume(__ESBMC_blocked_threads_count == 0);
  __ESBMC_atomic_end();

  // Ensure that there is no subsequent execution path
  __ESBMC_assume(0);
}
#pragma clang diagnostic pop

pthread_t pthread_self(void)
{
__ESBMC_HIDE:;
  return __ESBMC_get_thread_id();
}

/** Deliver a pending cancellation if cancelstate is PTHREAD_CANCEL_ENABLE. */
void pthread_testcancel(void)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t tid = __ESBMC_get_thread_id();
  _Bool should_cancel =
    __ESBMC_pthread_cancel_requested[tid] &&
    __ESBMC_pthread_cancelstate[tid] == PTHREAD_CANCEL_ENABLE;
  __ESBMC_atomic_end();
  if (should_cancel)
    pthread_exit(PTHREAD_CANCELED);
}

/** Send a cancellation request to thread th. */
int pthread_cancel(pthread_t th)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_pthread_cancel_requested[(int)th] = 1;
  __ESBMC_atomic_end();
  return 0;
}

/** Set the calling thread's cancellability state. */
int pthread_setcancelstate(int state, int *oldstate)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t tid = __ESBMC_get_thread_id();
  if (oldstate)
    *oldstate = __ESBMC_pthread_cancelstate[tid];
  __ESBMC_pthread_cancelstate[tid] = state;
  __ESBMC_atomic_end();
  return 0;
}

/** Set the calling thread's cancellation type. */
int pthread_setcanceltype(int type, int *oldtype)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t tid = __ESBMC_get_thread_id();
  if (oldtype)
    *oldtype = __ESBMC_pthread_canceltype[tid];
  __ESBMC_pthread_canceltype[tid] = type;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_join_switch(pthread_t thread, void **retval)
{
__ESBMC_HIDE:;
  pthread_testcancel();
  __ESBMC_atomic_begin();

  // Detect whether the target thread has ended or not. If it isn't, mark us as
  // waiting for its completion. That fact can be used for deadlock detection
  // elsewhere.
  _Bool ended = __ESBMC_pthread_thread_ended[(int)thread];
  if (!ended)
  {
    __ESBMC_blocked_threads_count++;
    // If there are now no more threads unblocked, croak.
    __ESBMC_assert(
      __ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
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

int pthread_join_noswitch(pthread_t thread, void **retval)
{
__ESBMC_HIDE:;
  //Tong: Dummy read, for recording mopr dependency of pthread_create()/join(), to avoid losing the dependency
  //relation when the cs swich is forced by pthread om. Should have a more elegant way to do.
  _Bool ended_sync = __ESBMC_pthread_thread_ended[(int)thread];

  pthread_testcancel();
  __ESBMC_atomic_begin();

  // If the other thread hasn't ended, assume false, because further progress
  // isn't going to be made. Wait for an interleaving where this is true
  // instead. This function isn't designed for deadlock detection.
  _Bool ended = __ESBMC_pthread_thread_ended[(int)thread];
  __ESBMC_assume(ended);

  // Fetch exit code
  if (retval != NULL)
    *retval = __ESBMC_pthread_end_values[(int)thread];

  __ESBMC_atomic_end();

  return 0;
}

/************************* Mutex manipulation routines ************************/

int pthread_mutex_init(
  pthread_mutex_t *mutex,
  const pthread_mutexattr_t *mutexattr)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_mutex_count_field(*mutex) = 0;
  __ESBMC_mutex_owner_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_initializer(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  // check whether this mutex has been initialized via
  // PTHREAD_MUTEX_INITIALIZER
  __ESBMC_atomic_begin();
  if (__ESBMC_mutex_lock_field(*mutex) == 0)
    pthread_mutex_init(mutex, NULL);
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_lock_noassert(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_mutex_initializer(mutex);
  __ESBMC_assume(!__ESBMC_mutex_lock_field(*mutex));
  __ESBMC_mutex_lock_field(*mutex) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_lock_nocheck(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_mutex_initializer(mutex);
  __ESBMC_assume(!__ESBMC_mutex_lock_field(*mutex));
  __ESBMC_mutex_lock_field(*mutex) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_unlock_noassert(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_unlock_nocheck(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(
    __ESBMC_mutex_lock_field(*mutex), "must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_lock_check(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  _Bool unlocked = 1;

  __ESBMC_atomic_begin();

  pthread_mutex_initializer(mutex);

  unlocked = (__ESBMC_mutex_lock_field(*mutex) == 0);

  if (unlocked)
  {
    __ESBMC_mutex_lock_field(*mutex) = 1;
  }
  else
  {
    // Deadlock foo
    __ESBMC_blocked_threads_count++;
    // No more threads to run -> croak.
    __ESBMC_assert(
      __ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
      "Deadlocked state in pthread_mutex_lock");
  }

  // Switch away for deadlock detection and so forth...
  __ESBMC_atomic_end();

  // ... but don't allow execution further if it was locked.
  __ESBMC_assume(unlocked);

  return 0;
}

int pthread_mutex_unlock_check(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(
    __ESBMC_mutex_lock_field(*mutex), "must hold lock upon unlock");
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  int res = EBUSY;
  if (__ESBMC_mutex_lock_field(*mutex) != 0)
    goto PTHREAD_MUTEX_TRYLOCK_END;

  pthread_mutex_lock(mutex);
  res = 0;

PTHREAD_MUTEX_TRYLOCK_END:
  __ESBMC_atomic_end();
  return res;
}

// The pthread_mutex_destroy() function shall destroy
// the mutex object referenced by mutex;
// the mutex object becomes, in effect, uninitialized.
int pthread_mutex_destroy_check(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(
    __ESBMC_mutex_lock_field(*mutex) != 0, "attempt to destroy a locked mutex");
  // Attempting to destroy a locked mutex results in undefined behavior.
  __ESBMC_assert(
    __ESBMC_mutex_lock_field(*mutex) == 1, "attempt to destroy a locked mutex");
  __ESBMC_assert(
    __ESBMC_mutex_lock_field(*mutex) != -1,
    "attempt to destroy a previously destroyed mutex");

  // It shall be safe to destroy an initialized mutex that is unlocked
  __ESBMC_mutex_lock_field(*mutex) = -1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_mutex_lock_field(*mutex) = -1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_destroy(pthread_rwlock_t *lock)
{
  return 0;
}

/************************ rwlock mainpulation routines ************************/

int pthread_rwlock_init(
  pthread_rwlock_t *lock,
  const pthread_rwlockattr_t *attr)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_rwlock_readers(lock) = 0;
  __ESBMC_rwlock_writer(lock) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_rdlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_rwlock_writer(lock));
  __ESBMC_rwlock_readers(lock)++;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_tryrdlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  if (__ESBMC_rwlock_writer(lock) != 0)
  {
    __ESBMC_atomic_end();
    return EBUSY;
  }
  else
    __ESBMC_rwlock_readers(lock)++;
  __ESBMC_atomic_end();

  return 0;
}

int pthread_rwlock_trywrlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  if (__ESBMC_rwlock_writer(lock) != 0 || __ESBMC_rwlock_readers(lock) != 0)
  {
    __ESBMC_atomic_end();
    return EBUSY;
  }
  __ESBMC_rwlock_writer(lock) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_unlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  if (__ESBMC_rwlock_writer(lock))
    __ESBMC_rwlock_writer(lock) = 0;
  else if (__ESBMC_rwlock_readers(lock) > 0)
    __ESBMC_rwlock_readers(lock)--;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_wrlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(
    !(__ESBMC_rwlock_writer(lock) || __ESBMC_rwlock_readers(lock)));
  __ESBMC_rwlock_writer(lock) = 1;
  __ESBMC_atomic_end();
  return 0;
}

/************************ condvar mainpulation routines ***********************/

// The pthread_cond_broadcast() function shall unblock
// all threads currently blocked on the specified condition
// variable cond.
int pthread_cond_broadcast(pthread_cond_t *cond)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*cond) = 0;
  __ESBMC_atomic_end();

  return 0;
}

int pthread_cond_init(
  pthread_cond_t *cond,
  __const pthread_condattr_t *cond_attr)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*cond) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_cond_destroy(pthread_cond_t *__cond)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*__cond) = 0;
  __ESBMC_atomic_end();
  return 0;
}

extern int pthread_cond_signal(pthread_cond_t *__cond)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_cond_lock_field(*__cond) = 0;
  __ESBMC_atomic_end();
  return 0;
}

static void
do_pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex, _Bool assrt)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  {
    pthread_t _ctid = __ESBMC_get_thread_id();
    _Bool _cancel = __ESBMC_pthread_cancel_requested[_ctid] &&
                    __ESBMC_pthread_cancelstate[_ctid] == PTHREAD_CANCEL_ENABLE;
    if (_cancel)
      __ESBMC_mutex_lock_field(*mutex) = 0;
    __ESBMC_atomic_end();
    if (_cancel)
      pthread_exit(PTHREAD_CANCELED);
  }
  __ESBMC_atomic_begin();

  if (assrt)
    __ESBMC_assert(
      __ESBMC_mutex_lock_field(*mutex),
      "caller must hold pthread mutex lock in pthread_cond_wait");

  // Unlock mutex; register us as waiting on condvar; context switch
  __ESBMC_mutex_lock_field(*mutex) = 0;
  __ESBMC_cond_lock_field(*cond) = 1;

  // Technically in the gap below, we are blocked. So mark ourselves thus. If
  // all other threads are (or become) blocked, then deadlock occurred, which
  // this helps detect.
  __ESBMC_blocked_threads_count++;
  // No more threads to run -> croak.
  if (assrt)
    __ESBMC_assert(
      __ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
      "Deadlocked state in pthread_mutex_lock");

  __ESBMC_atomic_end();

  // Other thread activity to happen in this gap

  __ESBMC_atomic_begin();

  // Have we been signalled?
  _Bool signalled = __ESBMC_cond_lock_field(*cond) == 0;

  /**
  * NOTE:
  *
  * When using condition variables there is always a boolean predicate
  * involving shared variables associated with each condition wait that
  * is true if the thread should proceed.
  *
  * Spurious wakeups from the pthread_cond_wait() or pthread_cond_timedwait()
  * functions may occur. Since the return from pthread_cond_wait() or pthread_cond_timedwait()
  * does not imply anything about the value of this predicate, the predicate should be re-evaluated
  * upon such return.
  *
  */
  _Bool spurious_wakeup = nondet_bool();
  signalled |= spurious_wakeup;

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

int pthread_cond_wait_nocheck(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  do_pthread_cond_wait(cond, mutex, 0);
  return 0;
}

int pthread_cond_wait_check(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
__ESBMC_HIDE:;
  do_pthread_cond_wait(cond, mutex, 1);
  return 0;
}

/************************ key mainpulation routines ***********************/

// The pthread_key_create() function shall create a thread-specific
// data key visible to all threads in the process.
// source: https://linux.die.net/man/3/pthread_key_create
int pthread_key_create(pthread_key_t *key, void (*destructor)(void *))
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(
    key != NULL,
    "In pthread_key_create, key parameter must be different than NULL.");
  // the value NULL shall be associated with the new key in all active threads
  insert_key_value(__ESBMC_next_thread_key, NULL);
  __ESBMC_thread_key_destructors[__ESBMC_next_thread_key] = destructor;
  // store the newly created key value at *key
  *key = __ESBMC_next_thread_key++;
  __ESBMC_atomic_end();
  return 0;
}

// The pthread_getspecific() function shall return
// the value currently bound to the specified key
// on behalf of the calling thread.
// source: https://linux.die.net/man/3/pthread_getspecific
void *pthread_getspecific(pthread_key_t key)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  // If no thread-specific data value is associated with key,
  // then the value NULL shall be returned.
  void *result = NULL;
  if (key <= __ESBMC_next_thread_key)
  {
    // Return the thread-specific data value associated
    // with the given key.
    __ESBMC_thread_key *l = search_key(key);
    result = (l == NULL) ? NULL : (void *)l->value;
  }
  __ESBMC_atomic_end();
  // No errors are returned from pthread_getspecific().
  return result;
}

// The pthread_setspecific() function shall associate
// a thread-specific value with a key obtained via
// a previous call to pthread_key_create().
// source: https://linux.die.net/man/3/pthread_setspecific
int pthread_setspecific(pthread_key_t key, const void *value)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  insert_key_value(key, value);
  // The key value is invalid when value is NULL.
  int result = (value == NULL) ? EINVAL : 0;
  __ESBMC_atomic_end();
  return result;
}

// The pthread_equal() function compares two thread identifiers.
// source: https://man7.org/linux/man-pages/man3/pthread_equal.3.html
int pthread_equal(pthread_t tid1, pthread_t tid2)
{
__ESBMC_HIDE:;
  // If the two thread IDs are equal,
  // it returns a nonzero value;
  // otherwise, it returns 0.
  __ESBMC_atomic_begin();
  _Bool res = tid1 == tid2;
  __ESBMC_atomic_end();
  // This function always succeeds.
  return res;
}

/************************ detach routine ***********************/

// The pthread_detach() function marks the thread identified by thread
// as detached.  When a detached thread terminates, its resources are
// automatically released back to the system without the need for
// another thread to join with the terminated thread.
// source: https://man7.org/linux/man-pages/man3/pthread_detach.3.html
int pthread_detach(pthread_t threadid)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  int result = 0;
  // This assert also checks whether this thread is not a joinable thread.
  __ESBMC_assert(
    !__ESBMC_pthread_thread_detach[threadid],
    "Attempting to detach an already detached thread results in unspecified "
    "behavior");
  if (
    __ESBMC_pthread_thread_ended[threadid] ||
    threadid > __ESBMC_num_total_threads)
    result = ESRCH; // No thread with the ID thread could be found.
  else
    __ESBMC_pthread_thread_detach[threadid] = 1;
  __ESBMC_atomic_end();
  return result; // no error occurred
}

/**
 * Push a cleanup handler onto the current thread's cleanup stack.
 *
 * This function registers a cleanup handler function along with its argument
 * to be called later if pthread_cleanup_pop is invoked with execute != 0.
 *
 * The cleanup handlers are stored in a large symbolic array divided into
 * chunks of size CLEANUP_STACK_CHUNK_SIZE per thread to avoid overlap between threads.
 * 
 * @param function Pointer to the cleanup function to be called.
 * @param arg      Argument to pass to the cleanup function.
 */
void pthread_cleanup_push(void (*function)(void *), void *arg)
{
  // Enable calling pthread_cleanup_pop from pthread_exit
  pthread_cleanup_enabled = 1;

  // Get the current thread ID
  pthread_t tid = __ESBMC_get_thread_id();

  // Get the current cleanup stack level for this thread
  size_t cleanup_level = __esbmc_get_cleanup_level();

  // Calculate the index for the cleanup entry in the symbolic infinite array.
  // Each thread gets a separate chunk of CLEANUP_STACK_CHUNK_SIZE slots to avoid interference.
  // Within the chunk, cleanup_level indexes the next free slot.
  size_t index = tid * CLEANUP_STACK_CHUNK_SIZE + cleanup_level;

  // Store the cleanup function pointer and its argument at the calculated index
  __esbmc_cleanup_stack[index].function = (void *)function;
  __esbmc_cleanup_stack[index].arg = arg;

  // Increase the cleanup stack level for the thread
  __esbmc_set_cleanup_level(cleanup_level + 1);
}

/**
 * Pop a cleanup handler from the current thread's cleanup stack and optionally execute it.
 *
 * This function removes the most recently pushed cleanup handler.
 * If execute is non-zero, it calls the cleanup function with the stored argument.
 *
 * @param execute If non-zero, execute the popped cleanup handler.
 */
void pthread_cleanup_pop(int execute)
{
  // This checks for API contract violation (undefined behavior)
  __ESBMC_assert(pthread_cleanup_enabled, "API contract: push/pop must match");

  // Get the current thread ID
  pthread_t tid = __ESBMC_get_thread_id();

  // Get the current cleanup stack level
  size_t cleanup_level = __esbmc_get_cleanup_level();

  // Assume the cleanup level is positive (there is a cleanup handler to pop)
  __ESBMC_assume(cleanup_level > 0);

  // Decrement the cleanup level to point to the handler being popped
  cleanup_level--;

  // Update the cleanup level
  __esbmc_set_cleanup_level(cleanup_level);

  if (execute)
  {
    // Calculate the index in the infinite symbolic cleanup stack for this thread and level
    size_t index = tid * CLEANUP_STACK_CHUNK_SIZE + cleanup_level;

    // Retrieve the stored cleanup function and argument
    void (*function)(void *) =
      (void (*)(void *))__esbmc_cleanup_stack[index].function;
    void *arg = __esbmc_cleanup_stack[index].arg;

    // Assume the function pointer is not NULL before calling
    __ESBMC_assume(function != NULL);

    // Call the cleanup function with the provided argument
    function(arg);
  }
}

/* Python ``threading.Thread`` helpers.
 *
 * The Python frontend's AST rewrite of ``threading.Thread`` lowers
 * each construction site to a per-site ``void(void)`` trampoline (a
 * Python function reading its arguments from module-level globals)
 * plus calls to these three intrinsics:
 *
 *   1. The spawn itself uses ``__ESBMC_spawn_thread`` directly — its
 *      argument has to be a literal ``address_of`` of the trampoline
 *      symbol (see ``intrinsic_spawn_thread`` in
 *      ``builtin_functions/threads.cpp``), and the Python converter
 *      wraps the trampoline reference accordingly.
 *
 *   2. ``__pyt_init_tid(tid)`` sets the pthread bookkeeping for the
 *      freshly-spawned thread (running/ended flags, counter bumps) so
 *      ``__pyt_join`` and the deadlock detector see consistent state.
 *      Mirrors the bookkeeping ``pthread_create`` does after its own
 *      call to ``__ESBMC_spawn_thread``.
 *
 *   3. ``__pyt_terminate()`` marks the calling thread ended,
 *      decrements the running count, and cuts subsequent execution
 *      paths. The synthesised trampoline calls this as its last
 *      statement so ``__pyt_join`` can observe the ``ended`` flag.
 *
 *   4. ``__pyt_join(tid)`` blocks until the named thread is ended.
 *      Mirrors ``pthread_join_switch`` (deadlock-aware) so symex's
 *      interleaving search can find the right schedule.
 */

void __pyt_init_tid(unsigned int tid)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_num_total_threads++;
  __ESBMC_num_threads_running++;
  __ESBMC_pthread_thread_running[tid] = 1;
  __ESBMC_pthread_thread_ended[tid] = 0;
  __ESBMC_pthread_end_values[tid] = NULL;
  __ESBMC_atomic_end();
}

void __pyt_terminate(void)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  pthread_t threadid = __ESBMC_get_thread_id();
  __ESBMC_pthread_thread_ended[threadid] = 1;
  __ESBMC_num_threads_running--;
  // ``pthread_trampoline`` additionally assumes
  // ``__ESBMC_blocked_threads_count == 0`` to prune deadlocked
  // schedules from its own path; we deliberately omit that assume
  // here so the join-side deadlock detector
  // (``blocked_threads_count != num_threads_running`` in
  // ``__pyt_join``) is the sole oracle, avoiding a circular
  // dependency where the joiner's pending block forces the spawned
  // thread to stall in terminate.
  __ESBMC_terminate_thread();
  __ESBMC_atomic_end(); // unreachable; mirrors pthread_trampoline.
  __ESBMC_assume(0);
}

void __pyt_join(unsigned int thread)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  _Bool ended = __ESBMC_pthread_thread_ended[(int)thread];
  if (!ended)
  {
    __ESBMC_blocked_threads_count++;
    // Deadlock if every running thread is now blocked.
    __ESBMC_assert(
      __ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
      "Deadlocked state in __pyt_join");
  }
  __ESBMC_atomic_end();
  // Block this thread until the target has ended.
  __ESBMC_assume(ended);
}

/* threading.Lock deadlock-aware bookkeeping.
 *
 * Called from ``Lock.acquire`` in the deadlock-aware variant of the
 * Python operational model (``models/threading_deadlock.py``) on the
 * branch where the lock is already held by another thread. Mirrors the
 * else-branch of ``pthread_mutex_lock_check``: bump the global blocked
 * counter and assert that not every running thread is now blocked. The
 * Python caller wraps this call in its own ``__ESBMC_atomic_begin`` /
 * ``__ESBMC_atomic_end`` pair, so the bump-and-assert is atomic with
 * the lock-field read that decided to block.
 *
 * The counter bump persists on a path that the caller then kills with
 * ``__ESBMC_assume(unlocked)``; symex's interleaving search observes
 * the bumped value on alternative schedules where two or more threads
 * have all reached this point, which is where the global deadlock
 * predicate (``blocked_threads_count == num_threads_running``) becomes
 * true.
 */
void __ESBMC_pylock_block_and_check(void)
{
__ESBMC_HIDE:;
  __ESBMC_blocked_threads_count++;
  __ESBMC_assert(
    __ESBMC_blocked_threads_count != __ESBMC_num_threads_running,
    "Deadlocked state in threading.Lock.acquire");
}
