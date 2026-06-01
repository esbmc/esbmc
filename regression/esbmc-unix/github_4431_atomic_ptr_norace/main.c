// Regression for issue #4431 (SV-COMP no-data-race false alarm on
// c/weaver/popl20-min-max-inc-dec.wvr).
//
// Two threads concurrently update the SAME object through a pointer to
// _Atomic int. Per C11 §5.1.2.4p4, concurrent accesses to an _Atomic object
// are not a data race, so this is race-free. ESBMC used to report a spurious
// W/W race because the rw_set #atomic filter only inspects the root symbol's
// own type: for `A[i]` with `_Atomic int *A`, neither the (non-atomic) pointer
// A nor the index i is itself atomic, so the access to the atomic element
// slipped past the filter. irep2 has no atomic-type node, so the fix recovers
// the flag from the base symbol's pointee/element type.
//
// Expected: VERIFICATION SUCCESSFUL.
#include <pthread.h>

_Atomic int storage;
_Atomic int *A;

void *t_inc(void *arg)
{
  A[0]++;
  return 0;
}
void *t_dec(void *arg)
{
  A[0]--;
  return 0;
}

int main(void)
{
  A = &storage;
  pthread_t x, y;
  pthread_create(&x, 0, t_inc, 0);
  pthread_create(&y, 0, t_dec, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  return 0;
}
