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
  return NULL;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, NULL, worker, NULL);
  g = 2;
  return 0;
}
