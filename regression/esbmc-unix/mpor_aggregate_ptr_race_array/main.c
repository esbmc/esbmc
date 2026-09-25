#include <pthread.h>
#include <assert.h>

int g = 0;
int *pa[1] = {&g};

// R29: a pointer in an array element. get_expr_globals gates its chain
// resolution on is_symbol2t, so this write reached the aggregate rather than g
// and MPOR pruned the race. The dereference2t arm resolves the pointer through
// the value set instead. The bare struct-member form is still open --
// see mpor_aggregate_ptr_race (KNOWNBUG).
void *writer(void *arg)
{
  (void)arg;
  *(pa[0]) = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
