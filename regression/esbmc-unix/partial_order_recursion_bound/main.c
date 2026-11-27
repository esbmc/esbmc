/* #10990: main's recursion is cut by its unwinding bound; past it main does
 * not run, so the check after the call is never reached. */
#include <pthread.h>
#include <assert.h>
int s;
void spin(void)
{
  spin();
}
void *t(void *arg)
{
  s++;
  assert(s <= 2);
  s--;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  spin();
  assert(s == 5);
  return 0;
}
