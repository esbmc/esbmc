#include <pthread.h>
pthread_rwlock_t A, B;

void *t1(void *p)
{
  pthread_rwlock_wrlock(&A);
  pthread_rwlock_wrlock(&B);
  pthread_rwlock_unlock(&B);
  pthread_rwlock_unlock(&A);
  return 0;
}

void *t2(void *p)
{
  pthread_rwlock_wrlock(&B);
  pthread_rwlock_wrlock(&A);
  pthread_rwlock_unlock(&A);
  pthread_rwlock_unlock(&B);
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_rwlock_init(&A, 0);
  pthread_rwlock_init(&B, 0);
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
