/* #10990: returning from main must not assume false under --partial-order,
   or the claims of the threads symex runs after main would be vacuous. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *a)
{
  assert(x == 0);
  return 0;
}
int main()
{
  pthread_t a;
  pthread_create(&a, 0, t, 0);
  x = 1;
  return 0;
}
