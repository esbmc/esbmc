// Companion to the indirect-write gating: a never-address-taken read-only
// global stays optimisable even when the program also performs an unresolved
// pointer write elsewhere.
//
// `*q = 9` is a dereference write, so the static scan sets any_indirect_write.
// The refined gating does NOT let that disable the optimisation globally: it is
// gated per-global on address_taken_globals. `ro`'s address is never taken, so
//   may_be_written(ro) == (any_indirect_write && 0) == false
// and ro's reads stay filtered. The pointer write targets `owned` (address-
// taken, written only by main before any spawn), which cannot race. The
// workers only read ro and each writes its own disjoint global, so the program
// is race-free: --data-races-check must report VERIFICATION SUCCESSFUL.
//
// (Observably, the flag still prunes ro's read interleavings here -- 161 -> 23
// on a read-heavy variant -- confirming ro remained filterable despite the
// indirect write.)
#include <pthread.h>
#include <assert.h>

int ro = 5;         // never address-taken, read-only across threads
int owned;          // address-taken, written only by main before spawn
int *q = &owned;
int out_a;          // written only by worker_a
int out_b;          // written only by worker_b

void *worker_a(void *_)
{
  out_a = ro + 1;
  return 0;
}

void *worker_b(void *_)
{
  out_b = ro + 2;
  return 0;
}

int main(void)
{
  *q = 9; // unresolved pointer write -> any_indirect_write = true
  pthread_t a, b;
  pthread_create(&a, 0, worker_a, 0);
  pthread_create(&b, 0, worker_b, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(out_a == 6);
  assert(out_b == 7);
  return 0;
}
