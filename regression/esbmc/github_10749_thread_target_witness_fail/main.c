// #10749: the target runs in the spawned thread, so it needs thread_id 1.
#include <assert.h>
#include <pthread.h>
int g;
void *t(void *arg)
{
  for (int k = 0; k < 2; k++)
    g++;
  assert(g != 2);
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
