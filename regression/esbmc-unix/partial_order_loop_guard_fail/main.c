/* t2 enters its loop if it reads t1's first write. Symex runs t1 first, and
 * t1 ends with flag cleared; t2's loop guard must not take that value. */
#include <pthread.h>
#include <assert.h>
int flag;
void *t1(void *a)
{
  flag = 1;
  flag = 0;
  return 0;
}
void *t2(void *a)
{
  while (flag >= 1)
    assert(0);
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  return 0;
}
