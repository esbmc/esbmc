#include <assert.h>
#include <pthread.h>

pthread_key_t k;
int a;
int calls = 0;

void dtor(void *p)
{
  ++calls;
  assert(calls != 2);
  if (calls == 1)
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
  return 0;
}
