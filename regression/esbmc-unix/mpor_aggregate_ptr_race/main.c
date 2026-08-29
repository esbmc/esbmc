#include <pthread.h>
#include <assert.h>

int g = 0;
struct S
{
  int *p;
};
struct S s = {&g};

// R29: writes g through a pointer held in a struct member. Constant
// propagation leaves `s` a constant_struct2t, whose members' values sit in the
// expression itself, so the suffixed symbol `c:@s.p` the value set is keyed on
// has nothing to match -- the write resolved to no object and MPOR pruned the
// race. get_value_set_rec now descends into the named member instead.
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
