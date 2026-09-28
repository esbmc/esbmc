#include <pthread.h>

static pthread_mutex_t m;

static void *w(void *p)
{
  (void)p;
  pthread_mutex_lock(&m);
  pthread_mutex_unlock(&m);
  return 0;
}

int main(void)
{
  pthread_t a;
  pthread_mutex_init(&m, 0);
  pthread_mutex_lock(&m);
  pthread_create(&a, 0, w, 0);
  pthread_mutex_init(&m, 0);
  pthread_join(a, 0);
  return 0;
}
