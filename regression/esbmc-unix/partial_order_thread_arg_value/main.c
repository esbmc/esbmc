/* #10990: each thread receives its argument's value at the spawn. */
#include <pthread.h>
#include <assert.h>
int a, b;
void *t(void *arg)
{
  if ((long)arg == 1)
    a = 1;
  else
    b = 2;
  return 0;
}
int main()
{
  pthread_t id1, id2;
  long k = 1;
  pthread_create(&id1, 0, t, (void *)k);
  k = 2;
  pthread_create(&id2, 0, t, (void *)k);
  pthread_join(id1, 0);
  pthread_join(id2, 0);
  assert(a == 1 && b == 2);
  return 0;
}
