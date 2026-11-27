/* #10990: the lost update needs both loops unwound, so the base case finds
 * it only at the bound where the joins return. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  for (int i = 0; i < 1; ++i)
  {
    int r = x;
    x = r + 1;
  }
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
