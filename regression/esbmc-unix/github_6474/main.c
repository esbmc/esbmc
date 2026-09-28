#include <pthread.h>
pthread_mutex_t m1, m2;

void *w(void *p)
{
  pthread_mutex_lock(&m2);
  pthread_mutex_lock(&m1);
  pthread_mutex_unlock(&m1);
  pthread_mutex_unlock(&m2);
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_mutex_init(&m1, 0);
  pthread_mutex_init(&m2, 0);
  pthread_mutex_lock(&m1);
  pthread_create(&a, 0, w, 0);
  pthread_create(&b, 0, w, 0);
  pthread_mutex_unlock(&m1);
  pthread_mutex_lock(&m2);
  pthread_mutex_unlock(&m2);
  return 0;
}
