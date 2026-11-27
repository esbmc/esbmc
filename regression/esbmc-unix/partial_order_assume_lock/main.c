/* #10990: a lock built from an assumption in an atomic section, as in
 * SV-COMP's assume_abort_if_not. The thread's assumption comes after main's
 * claim in the equation but still keeps the thread out of main's section. */
#include <pthread.h>
#include <assert.h>
int lock, x;
void acquire()
{
  __ESBMC_atomic_begin();
  __ESBMC_assume(lock == 0);
  lock = 1;
  __ESBMC_atomic_end();
}
void release()
{
  __ESBMC_atomic_begin();
  lock = 0;
  __ESBMC_atomic_end();
}
void *t(void *arg)
{
  acquire();
  x = 1;
  x = 0;
  release();
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  acquire();
  assert(x == 0);
  release();
  return 0;
}
