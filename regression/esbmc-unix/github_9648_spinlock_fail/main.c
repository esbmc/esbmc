/* #9648: relocking the spinlock blocks forever, so only x == 0 continues. */
#include <assert.h>
#include <pthread.h>

int nondet_int(void);
pthread_spinlock_t s;

int main(void)
{
  int x = nondet_int();
  pthread_spin_init(&s, 0);
  pthread_spin_lock(&s);
  if (x)
    pthread_spin_lock(&s);
  assert(x != 0);
  return 0;
}
