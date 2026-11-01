int __VERIFIER_nondet_int(void);
#include <assert.h>
#include <pthread.h>

pthread_spinlock_t s;

int main(void)
{
  pthread_spin_init(&s, 0);
  pthread_spin_lock(&s);
  if (__VERIFIER_nondet_int())
    pthread_spin_lock(&s);
  return 0;
}
