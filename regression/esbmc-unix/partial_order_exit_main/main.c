/* #10990: main ending with pthread_exit lets its thread run on, and main's
   code after the call never runs. */
#include <pthread.h>
#include <assert.h>
int x = 0;
void *t(void *a)
{
  assert(x == 1);
  return 0;
}
int main()
{
  pthread_t id;
  x = 1;
  pthread_create(&id, 0, t, 0);
  pthread_exit(0);
  x = 2;
  return 0;
}
