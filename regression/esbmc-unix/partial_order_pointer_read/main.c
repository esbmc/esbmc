/* #10990: a read through a pointer to a global sees the other thread's write
   once that thread is joined. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t id;
  int *p = &x;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  assert(*p == 1);
  return 0;
}
