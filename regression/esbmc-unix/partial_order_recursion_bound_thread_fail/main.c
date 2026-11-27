/* #10990: a recursion cut in t1 must not hide t2's view of t1's earlier
 * write, though t1 comes first in the equation. */
#include <pthread.h>
#include <assert.h>
int x;
void spin(void)
{
  spin();
}
void *t1(void *arg)
{
  x = 1;
  spin();
  return 0;
}
void *t2(void *arg)
{
  assert(x == 0);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
}
