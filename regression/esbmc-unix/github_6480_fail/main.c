/* Companion to github_6480: the same array of locks, taken in opposite
   orders by the two threads. Pins that filtering mutex-array accesses out of
   context-switch generation does not cost the deadlock -- it must still be
   found. */
#include <pthread.h>
pthread_mutex_t m[2];

void *w1(void *a)
{
  pthread_mutex_lock(&m[0]);
  pthread_mutex_lock(&m[1]);
  pthread_mutex_unlock(&m[1]);
  pthread_mutex_unlock(&m[0]);
  return 0;
}

void *w2(void *a)
{
  pthread_mutex_lock(&m[1]);
  pthread_mutex_lock(&m[0]);
  pthread_mutex_unlock(&m[0]);
  pthread_mutex_unlock(&m[1]);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_mutex_init(&m[0], 0);
  pthread_mutex_init(&m[1], 0);
  pthread_create(&t1, 0, w1, 0);
  pthread_create(&t2, 0, w2, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
