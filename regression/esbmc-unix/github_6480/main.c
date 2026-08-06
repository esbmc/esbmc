/* Both threads take the array locks in the same order, so there is no
   deadlock. The point of the test is the cost of reaching that verdict:
   `pthread_mutex_t m[2]` collects as the array symbol, whose type is an
   array rather than the mutex struct, so before #6480 every acquisition
   slipped past the sync-type filter in has_cswitch_point_occured and forced
   a context-switch point -- 34168 interleavings here versus 10358 once the
   filter looks through the array, and versus 4764 for the same program
   written with two scalar mutexes. */
#include <pthread.h>
pthread_mutex_t m[2];

void *w(void *a)
{
  pthread_mutex_lock(&m[0]);
  pthread_mutex_lock(&m[1]);
  pthread_mutex_unlock(&m[1]);
  pthread_mutex_unlock(&m[0]);
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_mutex_init(&m[0], 0);
  pthread_mutex_init(&m[1], 0);
  pthread_create(&t1, 0, w, 0);
  pthread_create(&t2, 0, w, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  return 0;
}
