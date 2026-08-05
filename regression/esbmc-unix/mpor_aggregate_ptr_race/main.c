#include <pthread.h>
#include <assert.h>

int g = 0;
struct S
{
  int *p;
};
struct S s = {&g};

// Writes g through a pointer held in a struct member. get_expr_globals gates
// its pointer-chain resolution on is_symbol2t, so `s.p` -- a member2t -- never
// enters it: the write is keyed on `s` while main keys on `g`, MPOR calls the
// two transitions independent and prunes the racy interleaving (R28). Any
// aggregate step between the pointer and its name does this; see
// mpor_aggregate_ptr_race_local for the shape that still works.
void *writer(void *arg)
{
  (void)arg;
  *(s.p) = 1;
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
