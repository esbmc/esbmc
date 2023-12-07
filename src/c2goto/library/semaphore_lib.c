#include <semaphore.h>

#define __ESBMC_sem_lock_field(a) ((a).__lock)
#define __ESBMC_sem_count_field(a) ((a).__count)
#define __ESBMC_sem_init_field(a) ((a).__init)

/************************* Sem manipulation routines ************************/

int sem_init(sem_t *__sem, int __pshared, unsigned int __value)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  __ESBMC_assert(!__pshared, "ESBMC does not check multiprocesses for now");
  __ESBMC_sem_lock_field(*__sem) = 0;
  __ESBMC_sem_count_field(*__sem) = __value;
  __ESBMC_sem_init_field(*__sem) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int sem_init_check(sem_t *__sem)
{
  // check whether this sem has been initialised
  __ESBMC_atomic_begin();
  __ESBMC_assert(__ESBMC_sem_init_field(*__sem), "Sem is not initialised");
  __ESBMC_atomic_end();
  return 0;
}

int sem_wait_check(sem_t *__sem)
{
__ESBMC_HIDE:;
  __ESBMC_atomic_begin();
  sem_init_check(__sem);
  __ESBMC_sem_count_field(*__sem) -= 1;
  __ESBMC_assert(
    __ESBMC_sem_count_field(*__sem) < 0, "Deadlocked state in sem_wait");
  __ESBMC_assume(!__ESBMC_sem_lock_field(*__sem));
  if (!__ESBMC_sem_count_field(*__sem))
    __ESBMC_sem_lock_field(*__sem) = 1;
  __ESBMC_atomic_end();
  return 0;
}

int sem_wait_noassert(sem_t *__sem)
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

int sem_post_check(sem_t *__sem)
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

int sem_post_noassert(sem_t *__sem)
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

int sem_post_nocheck(sem_t *__sem)
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
