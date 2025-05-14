#include <errno.h>
#include <pthread.h>
#include <stddef.h>

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
#define __ESBMC_rwlock_field(a) ((a).__lock)

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

pthread_t __ESBMC_get_thread_id(void);

void __ESBMC_really_atomic_begin(void);
void __ESBMC_really_atomic_end(void);

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

static int insert_key_value(pthread_key_t key, const void *value)
{
__ESBMC_HIDE:;
  pthread_t thread = __ESBMC_get_thread_id();
  __ESBMC_pthread_thread_key[thread].thread = thread;
  __ESBMC_pthread_thread_key[thread].key = key;
  __ESBMC_pthread_thread_key[thread].value = value;
  return 0;
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
  __ESBMC_pthread_end_values[(int)threadid] = exit_val;
  __ESBMC_pthread_thread_ended[(int)threadid] = 1;
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
  pthread_exec_key_destructors();
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

  // Ensure that there is no subsequent execution path
  __ESBMC_assume(0);
}
#pragma clang diagnostic pop

pthread_t pthread_self(void)
{
__ESBMC_HIDE:;
  return __ESBMC_get_thread_id();
}

int pthread_join_switch(pthread_t thread, void **retval)
{
__ESBMC_HIDE:;
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
  __ESBMC_rwlock_field(*lock) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_rdlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_rwlock_field(*lock));
  __ESBMC_rwlock_field(*lock) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_tryrdlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  return 0;
}

int pthread_rwlock_trywrlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();

  int res = 1;
  if (__ESBMC_rwlock_field(*lock))
    goto PTHREAD_RWLOCK_TRYWRLOCK_END;

  __ESBMC_rwlock_field(*lock) = 1;
  res = 0;

PTHREAD_RWLOCK_TRYWRLOCK_END:
  __ESBMC_atomic_end();
  return res;
}

int pthread_rwlock_unlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_rwlock_field(*lock) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int pthread_rwlock_wrlock(pthread_rwlock_t *lock)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assume(!__ESBMC_rwlock_field(*lock));
  __ESBMC_rwlock_field(*lock) = 1;
  __ESBMC_atomic_end();
  return 0; // we never fail
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
  int result = insert_key_value(__ESBMC_next_thread_key, NULL);
  __ESBMC_thread_key_destructors[__ESBMC_next_thread_key] = destructor;
  // store the newly created key value at *key
  *key = __ESBMC_next_thread_key++;
  // check whether we have failed to insert the key into our list.
  if (result < 0)
  {
    if (nondet_bool())
    {
      // Insufficient memory exists to create the key.
      result = ENOMEM;
    }
    else
    {
      // The system lacked the necessary resources
      // to create another thread-specific data key, or
      // the system-imposed limit on the total number of
      // keys per process {PTHREAD_KEYS_MAX} has been exceeded.
      result = EAGAIN;
    }
  }
  __ESBMC_atomic_end();
  return result;
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
  int result;
  __ESBMC_atomic_begin();
  result = insert_key_value(key, value);
  if (result < 0)
  {
    // Insufficient memory exists to associate
    // the value with the key.
    result = ENOMEM;
  }
  else if (value == NULL)
  {
    // The key value is invalid.
    result = EINVAL;
  }
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
    !__ESBMC_pthread_thread_detach[(int)threadid],
    "Attempting to detach an already detached thread results in unspecified "
    "behavior");
  if (
    __ESBMC_pthread_thread_ended[(int)threadid] ||
    (int)threadid > __ESBMC_num_total_threads)
    result = ESRCH; // No thread with the ID thread could be found.
  else
    __ESBMC_pthread_thread_detach[(int)threadid] = 1;
  __ESBMC_atomic_end();
  return result; // no error occurred
}
