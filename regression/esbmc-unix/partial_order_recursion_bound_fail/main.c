/* #10990: main's recursion is cut by its unwinding bound, which must stop
 * only main: the spawned threads still run and can see each other's s++. */
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
  assert(s == 1);
  s--;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  spin();
}
