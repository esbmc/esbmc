/* #10990: main's spawning loop is cut by its unwinding bound; past it main
 * does not run, so the check after the loop is never reached. */
#include <pthread.h>
#include <assert.h>
int s;
void *t(void *arg)
{
  s++;
  assert(s <= 2);
  s--;
  return 0;
}
int main()
{
  pthread_t id;
  int i = 0;
  while (i < 5)
  {
    pthread_create(&id, 0, t, 0);
    ++i;
  }
  assert(i == 5);
  return 0;
}
