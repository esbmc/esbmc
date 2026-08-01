/* Barrier of 3, but only 2 threads ever arrive: a real deadlock. */
#include <pthread.h>
pthread_barrier_t bar;

void *t(void *p) { pthread_barrier_wait(&bar); return 0; }

int main(void)
{
  pthread_t a, b;
  pthread_barrier_init(&bar, 0, 3);
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
