#include <assert.h>
#include <pthread.h>

pthread_key_t k;
int a;
int cleanup_ran = 0;

void dtor(void *p)
{
  assert(!cleanup_ran);
}

void handler(void *p)
{
  cleanup_ran = 1;
}

void *t(void *arg)
{
  pthread_setspecific(k, &a);
  pthread_cleanup_push(handler, 0);
  pthread_exit(0);
  pthread_cleanup_pop(0);
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
