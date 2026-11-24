/* #10990: spawn and join events name the thread, and only the user's atomic
   section is an event, not the operational model's. */
#include <pthread.h>
int x;
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  x = 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
