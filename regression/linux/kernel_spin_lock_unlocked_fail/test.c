#include <ubuntu20.04/kernel_5.15.0-76/include/linux/spinlock.h>
#include <assert.h>
#include <pthread.h>

/* Control for kernel_spin_lock_blocks: drop the lock and the increments race,
   so the same assertion must be reachable. Keeps the pair discriminating -- a
   spin_lock that never excluded anything would pass both. */
int shared_counter = 0;
spinlock_t lock;
#define MAX_THREADS 2

void *increment(void *arg)
{
  int local = shared_counter;
  shared_counter = local + 1;
  return NULL;
}

int main()
{
  pthread_t thread1, thread2;
  spin_lock_init(&lock);
  pthread_create(&thread1, NULL, increment, NULL);
  pthread_create(&thread2, NULL, increment, NULL);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  assert(shared_counter == MAX_THREADS);
  return 0;
}
