#include <pthread.h>
#include <assert.h>
pthread_spinlock_t sl;
int shared = 0;

void *t(void *p)
{
  pthread_spin_lock(&sl);
  shared++;
  pthread_spin_unlock(&sl);
  return 0;
}

int main(void)
{
  pthread_t a;
  pthread_spin_init(&sl, 0);
  pthread_create(&a, 0, t, 0);
  pthread_spin_lock(&sl);
  shared++;
  pthread_spin_unlock(&sl);
  pthread_join(a, 0);
  assert(shared == 2);
  pthread_spin_destroy(&sl);
  return 0;
}
