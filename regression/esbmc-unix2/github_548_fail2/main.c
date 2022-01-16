#include <pthread.h>
#include <assert.h>

int glob1 = 0;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  pthread_mutex_lock(&mutex2);
  glob1 = 5;
  pthread_mutex_unlock(&mutex2);
  pthread_mutex_lock(&mutex2);
  glob1 = 10;
  pthread_mutex_unlock(&mutex2);
  return NULL;
}

// Race condition here
void *foo(void *arg)
{
  pthread_mutex_lock(&mutex2);
  assert(glob1 == 5 || glob1 == 10);
  pthread_mutex_unlock(&mutex2);
  return NULL;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, NULL, t_fun, NULL);
  foo(NULL);
  pthread_join(id, NULL);
  return 0;
}
