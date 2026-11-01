/* x == 1 is never visible outside main's critical section (#9653). */
#include <assert.h>
#include <pthread.h>
void reach_error(void) { assert(0); }
pthread_mutex_t s;
int x;
void *t(void *a)
{
  pthread_mutex_lock(&s);
  if (x == 1)
    reach_error();
  pthread_mutex_unlock(&s);
  return 0;
}
int main(void)
{
  pthread_mutex_init(&s, 0);
  pthread_t th;
  pthread_create(&th, 0, t, 0);
  pthread_mutex_lock(&s);
  x = 1;
  x = 2;
  pthread_mutex_unlock(&s);
  pthread_join(th, 0);
  return 0;
}
