/* #10990: without the join, main may read x before t writes it through p. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  int *p = &x;
  *p = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x == 1);
  return 0;
}
