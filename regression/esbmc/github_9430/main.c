#include <assert.h>
#include <pthread.h>

pthread_key_t k;
int a;
int calls = 0;

void dtor(void *p)
{
  assert(pthread_getspecific(k) == 0);
  assert(p == &a);
  ++calls;
  pthread_setspecific(k, &a);
}

void *t(void *arg)
{
  pthread_setspecific(k, &a);
  return 0;
}

int main(void)
{
  pthread_t th;
  pthread_key_create(&k, dtor);
  pthread_create(&th, 0, t, 0);
  pthread_join(th, 0);
  assert(calls == 4);
  return 0;
}
