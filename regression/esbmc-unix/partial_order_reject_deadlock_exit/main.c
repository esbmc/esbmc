/* #10990: the deadlock claim takes main's end for the process's end, which
   pthread_exit breaks. */
#include <pthread.h>
void *t(void *arg)
{
  pthread_exit(0);
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
