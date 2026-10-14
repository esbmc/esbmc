/* Self-deadlock: a spinlock is not recursive. */
#include <pthread.h>
pthread_spinlock_t sl;
int main(void)
{
  pthread_spin_init(&sl, 0);
  pthread_spin_lock(&sl);
  pthread_spin_lock(&sl);
  return 0;
}
