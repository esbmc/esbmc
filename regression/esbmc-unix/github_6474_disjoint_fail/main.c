/* The nested counterpart of github_6474_disjoint: t1 holds m while waiting for
   n and t2 holds n while waiting for m, so a wait-for cycle does exist and the
   deadlock must still be reported. Pairs with that test so cancelling waiters
   at unlock time cannot be tightened into missing real deadlocks. */
#include <pthread.h>

pthread_mutex_t m, n;

void *t1(void *a)
{
  pthread_mutex_lock(&m);
  pthread_mutex_lock(&n);
  pthread_mutex_unlock(&n);
  pthread_mutex_unlock(&m);
  return 0;
}

void *t2(void *a)
{
  pthread_mutex_lock(&n);
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  pthread_mutex_unlock(&n);
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_mutex_init(&m, 0);
  pthread_mutex_init(&n, 0);
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
