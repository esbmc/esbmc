#include <pthread.h>
#include <assert.h>

int g = 0;
struct S
{
  int *p;
};
struct S s = {&g};

// The control for mpor_aggregate_ptr_race: copying the pointer into a local
// gives get_expr_globals a bare symbol to resolve, so the write is keyed on g
// and MPOR keeps the interleaving. The two differ only in that copy, which is
// what pins R29 to the aggregate step rather than to the struct.
void *writer(void *arg)
{
  (void)arg;
  int *lp = s.p;
  *lp = 1;
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
