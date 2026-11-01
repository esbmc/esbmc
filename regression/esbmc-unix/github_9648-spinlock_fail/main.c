/* Relocking a held spinlock never returns, so it deadlocks. */
#include <assert.h>
#include <pthread.h>

pthread_spinlock_t sl;

int main(void)
{
  pthread_spin_init(&sl, 0);
  pthread_spin_lock(&sl);
  if (__VERIFIER_nondet_int())
  {
    pthread_spin_lock(&sl);
    assert(0);
  }
  return 0;
}
