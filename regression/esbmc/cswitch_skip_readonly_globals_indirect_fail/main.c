// Soundness guard for the indirect-write gating in may_be_written():
//   any_indirect_write && address_taken_globals.count(name)
//
// `g` is never the target of a *named* store -- it is written only through the
// pointer `p`, an unresolved dereference write (`*p = 1`). The static scan sees
// the dereference, sets any_indirect_write, and does NOT add g to
// ever_written_globals. But g's address escapes via `int *p = &g`, so g is in
// address_taken_globals. The gating must therefore keep g race-eligible: its
// reads must still trigger context switches.
//
// The reader loads g twice; a context switch to the writer *between* the two
// loads makes them disagree (a=0, b=1). With the gating working and POR off,
// that interleaving is explored and assert(a == b) fails -> VERIFICATION
// FAILED. If the gating wrongly classified g as read-only, the reader's loads
// would be filtered, no switch would be inserted between them, and the race
// would be hidden (spurious SUCCESSFUL).
#include <pthread.h>
#include <assert.h>

int g;
int *p = &g; // g's address escapes -> g is address-taken

void *writer(void *_)
{
  *p = 1; // unresolved pointer write -> any_indirect_write, target is g
  return 0;
}

void *reader(void *_)
{
  int a = g;
  int b = g;
  assert(a == b); // fails when the writer runs between the two loads
  return 0;
}

int main(void)
{
  pthread_t w, r;
  pthread_create(&w, 0, writer, 0);
  pthread_create(&r, 0, reader, 0);
  pthread_join(w, 0);
  pthread_join(r, 0);
  return 0;
}
