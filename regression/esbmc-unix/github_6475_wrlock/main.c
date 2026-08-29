#include <pthread.h>

static pthread_rwlock_t a, b;

static void *t1(void *p)
{
  (void)p;
  pthread_rwlock_wrlock(&a);
  pthread_rwlock_wrlock(&b);
  pthread_rwlock_unlock(&b);
  pthread_rwlock_unlock(&a);
  return 0;
}

static void *t2(void *p)
{
  (void)p;
  pthread_rwlock_wrlock(&b);
  pthread_rwlock_wrlock(&a);
  pthread_rwlock_unlock(&a);
  pthread_rwlock_unlock(&b);
  return 0;
}

int main(void)
{
  pthread_t p1, p2;
  pthread_rwlock_init(&a, 0);
  pthread_rwlock_init(&b, 0);
  pthread_create(&p1, 0, t1, 0);
  pthread_create(&p2, 0, t2, 0);
  /* Hand the process over to the workers: returning from main would call
     exit() and tear them down before they can deadlock (#6479). */
  pthread_exit(0);
}
