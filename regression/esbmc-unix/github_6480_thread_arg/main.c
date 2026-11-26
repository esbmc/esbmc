#include <assert.h>
#include <pthread.h>

pthread_mutex_t m[3];
int count;

void *worker(void *arg)
{
  long n = (long)arg;
  for (long k = 0; k < n; k++)
  {
    pthread_mutex_lock(&m[k]);
    count++;
    pthread_mutex_unlock(&m[k]);
  }
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, worker, (void *)3);
  pthread_join(t, 0);
  assert(count == 3);
  return 0;
}
