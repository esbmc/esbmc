/* #10990: each shared read gets its own fresh SSA name, so two reads of x in
   one thread may observe different writes; a local is not an event. */
#include <pthread.h>
int x;
void *t(void *a)
{
  int l = x;
  int m = x;
  l = l + m;
  return 0;
}
int main()
{
  pthread_t a;
  pthread_create(&a, 0, t, 0);
  x = 1;
  return 0;
}
