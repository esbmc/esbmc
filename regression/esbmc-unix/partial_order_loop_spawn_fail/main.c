/* #10990: threads from one spawning loop interleave, so an increment split
 * into a read and a write can be lost. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  int r = x;
  x = r + 1;
  return 0;
}
int main()
{
  pthread_t id[3];
  for (int i = 0; i < 3; ++i)
    pthread_create(&id[i], 0, t, 0);
  for (int i = 0; i < 3; ++i)
    pthread_join(id[i], 0);
  assert(x == 3);
  return 0;
}
