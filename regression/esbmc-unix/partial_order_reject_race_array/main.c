/* #10990: an array is one object, so its elements cannot be checked apart. */
#include <pthread.h>
int a[2];
void *t(void *arg)
{
  a[1] = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  a[0] = 1;
  return 0;
}
