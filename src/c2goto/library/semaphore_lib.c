#include <semaphore.h>
#include <limits.h>
#include <pthread.h>

#define __ESBMC_sem_lock_field(a) ((a).__lock)
#define __ESBMC_sem_count_field(a) ((a).__count)
#define __ESBMC_sem_init_field(a) ((a).__init)

/* Declaration of external variables */
extern unsigned short int __ESBMC_num_threads_running;
extern unsigned short int __ESBMC_blocked_threads_count;

/************************* Sem manipulation routines ************************/

int sem_init(sem_t *__sem, int __pshared, unsigned int __value)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  // ESBMC does not support shared memory at the moment
  // we ignore the __pshared for now
  __ESBMC_assert(__value <= INT_MAX, "Value has exceeded the maximum");
  __ESBMC_sem_lock_field(*__sem) = 0;
  __ESBMC_sem_count_field(*__sem) = __value;
  __ESBMC_sem_init_field(*__sem) = 1;
  __ESBMC_atomic_end();
  return 0;
}

static int sem_init_check(sem_t *__sem)
{
__ESBMC_HIDE:;
  // check whether this sem has been initialized
  __ESBMC_atomic_begin();
  __ESBMC_assert(__ESBMC_sem_init_field(*__sem), "Sem is not initialized");
  __ESBMC_atomic_end();
  return 0;
}

int sem_wait_check(sem_t *__sem)
{
__ESBMC_HIDE:;
  _Bool unlocked = 1;

  __ESBMC_atomic_begin();
  sem_init_check(__sem);
  unlocked = (__ESBMC_sem_lock_field(*__sem) == 0);
  if (unlocked)
  {
    __ESBMC_sem_count_field(*__sem) -= 1;
    __ESBMC_assume(!__ESBMC_sem_lock_field(*__sem));
    if (!__ESBMC_sem_count_field(*__sem))
      __ESBMC_sem_lock_field(*__sem) = 1;
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
  __ESBMC_atomic_end();

  __ESBMC_assume(unlocked);
  return 0;
}

int sem_wait_nocheck(sem_t *__sem)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  sem_init_check(__sem);
  __ESBMC_sem_count_field(*__sem) -= 1;
  __ESBMC_assume(!__ESBMC_sem_lock_field(*__sem));
  if (!__ESBMC_sem_count_field(*__sem))
    __ESBMC_sem_lock_field(*__sem) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int sem_post(sem_t *__sem)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  sem_init_check(__sem);
  __ESBMC_sem_count_field(*__sem) += 1;
  if (__ESBMC_sem_count_field(*__sem))
    __ESBMC_sem_lock_field(*__sem) = 0;
  __ESBMC_atomic_end();
  return 0;
}

int sem_destroy(sem_t *__sem)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_sem_lock_field(*__sem) = -1;
  __ESBMC_atomic_end();
  return 0;
}
