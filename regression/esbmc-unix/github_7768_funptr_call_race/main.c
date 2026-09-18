#include <pthread.h>

/* Calling through the global `fp` reads shared memory, so the call was
 * instrumented inside an atomic block: bump() then ran atomically and the race
 * on `shared` inside it was never reported. */
int shared;
void (*fp)(void);

void bump(void)
{
  shared = shared + 1;
}

void *t(void *arg)
{
  fp();
  return 0;
}

int main(void)
{
  fp = bump;
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  return 0;
}
