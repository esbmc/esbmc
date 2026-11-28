/* Publishing the flag before the data is a real bug, still found with the
 * --smt-symex-* flags under --partial-order. */
#include <pthread.h>
#include <assert.h>
int ready, data;
void *producer(void *arg)
{
  ready = 1;
  data = 1;
  return 0;
}
int main()
{
  pthread_t p;
  pthread_create(&p, 0, producer, 0);
  if (ready)
    assert(data == 1);
  pthread_join(p, 0);
  return 0;
}
