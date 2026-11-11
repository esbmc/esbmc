#include <assert.h>
#include <pthread.h>
#include <threads.h>

int a, b, c;
pthread_key_t k1, k2;

void *worker(void *arg)
{
  assert(pthread_getspecific(k1) == NULL);
  pthread_setspecific(k1, &c);
  assert(pthread_getspecific(k1) == &c && pthread_getspecific(k2) == NULL);
  return NULL;
}

int main(void)
{
  pthread_key_create(&k1, 0);
  pthread_setspecific(k1, &a);
  pthread_key_create(&k2, 0);
  pthread_setspecific(k2, &b);
  assert(pthread_getspecific(k1) == &a && pthread_getspecific(k2) == &b);

  tss_t t1, t2;
  tss_create(&t1, 0);
  tss_create(&t2, 0);
  tss_set(t1, &b);
  tss_set(t2, &a);
  assert(tss_get(t1) == &b && tss_get(t2) == &a);

  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  pthread_join(t, 0);
  assert(pthread_getspecific(k1) == &a);
  return 0;
}
