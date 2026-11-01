/* The lock taken right after main releases it must see x == 1 (#9653). */
#include <assert.h>
#include <pthread.h>
void reach_error(void) { assert(0); }
pthread_spinlock_t s;
int x;
void *t(void *a)
{
  pthread_spin_lock(&s);
  if (x == 1)
    reach_error();
  pthread_spin_unlock(&s);
  return 0;
}
int main(void)
{
  pthread_spin_init(&s, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_spin_lock(&s);
  x = 1;
  pthread_spin_unlock(&s);
  x = 2;
  pthread_join(th, 0);
  return 0;
}
