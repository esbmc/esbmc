// #10749: a race's final segment holds the earlier access, so a branch fixed
// by the program text gets no waypoint after the spawn.
#include <pthread.h>
int g;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
int h;
void *t(void *arg)
{
  pthread_mutex_lock(&m);
  h = 1;
  pthread_mutex_unlock(&m);
  g = 1;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_mutex_lock(&m);
  int x = h;
  pthread_mutex_unlock(&m);
  if (x)
    x = 2;
  g = 2;
  pthread_join(id, 0);
  return 0;
}
