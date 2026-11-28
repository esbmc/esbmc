/* The --smt-symex-* questions are asked of the equation symex has built so
 * far, where threads run one after another; under --partial-order a read may
 * take any write, so those answers pruned feasible interleavings. */
#include <pthread.h>
#include <assert.h>
int ready, data;
void *producer(void *arg)
{
  data = 1;
  ready = 1;
  return 0;
}
int main()
{
  pthread_t p;
  pthread_create(&p, 0, producer, 0);
  if (ready)
    assert(data == 1);
  pthread_join(p, 0);
  assert(ready == 1 && data == 1);
  return 0;
}
