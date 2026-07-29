/* The two threads hold different elements of the same lock array, so they
   share nothing and are independent. MPOR keyed every access on the base
   symbol `m`, which made them look dependent: 4973 interleavings here versus
   732 for the same program written with two scalar mutexes. Refining a
   constant offset into the key (#6480) brings this to 2780. */
#include <pthread.h>
pthread_mutex_t m[2];

void *w1(void *a)
{
  pthread_mutex_lock(&m[0]);
  pthread_mutex_unlock(&m[0]);
  return 0;
}

void *w2(void *a)
{
  pthread_mutex_lock(&m[1]);
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
