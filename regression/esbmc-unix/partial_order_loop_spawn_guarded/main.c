/* #10990: a thread the loop does not reach writes nothing, and joins on
 * thread ids merged from both paths wait for the threads that ran. */
#include <pthread.h>
#include <assert.h>
int nondet_int();
int x;
void *t(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t id[2];
  int n = nondet_int();
  if (n < 0 || n > 2)
    n = 2;
  for (int i = 0; i < n; ++i)
    pthread_create(&id[i], 0, t, 0);
  for (int i = 0; i < n; ++i)
    pthread_join(id[i], 0);
  assert(x == (n > 0));
  return 0;
}
