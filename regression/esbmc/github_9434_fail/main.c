// #9434: a TSD destructor is application code; main can observe its
// intermediate state while the exiting thread runs it.
#include <assert.h>
#include <pthread.h>

pthread_key_t k;
int v, x = 0;

void dtor(void *p)
{
  x = 1;
  x = 0;
}

void *t(void *a)
{
  pthread_setspecific(k, &v);
  return 0;
}

int main(void)
{
  pthread_t th;
  pthread_key_create(&k, dtor);
  pthread_create(&th, 0, t, 0);
  assert(x != 1);
  pthread_join(th, 0);
  return 0;
}
