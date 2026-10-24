#include <assert.h>
#include <pthread.h>

int g;
pthread_once_t once = PTHREAD_ONCE_INIT;

void init(void)
{
  g = 1;
}

void *worker(void *arg)
{
  pthread_once(&once, init);
  assert(g == 1);
  return NULL;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  pthread_once(&once, init);
  assert(g == 1);
  return 0;
}
