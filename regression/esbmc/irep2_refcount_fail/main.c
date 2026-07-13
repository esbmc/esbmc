// H-A1 anti-vacuity variant — proves the harness DETECTS a refcount defect.
//
// Identical to irep2_refcount_ok/main.c except copy() omits the
// refcount.fetch_add(1) that every copy ctor / copy-assign performs
// (irep2.h:177,205). This models a dropped bump on the copy ctor: the
// node is then under-counted (live slots > refcount), which in the real
// container would let a drop free storage a live slot still points at
// (early-free / use-after-free).
//
// Observed here: the under-count trips the `live == refcount` invariant
// (I1) at the end of the first copying step, *before* any premature free
// is reached -> VERIFICATION FAILED, demonstrating the passing harness
// has discriminating power. The test.desc pins this specific I1 violation.

#include <assert.h>

#define N 3
#define K 5

int nondet_int(void);
_Bool nondet_bool(void);

unsigned refcount = 1;
int deleted = 0;
_Bool slot_live[N];

void copy(int src, int dst)
{
  if (!slot_live[src] || slot_live[dst])
    return;
  slot_live[dst] = 1;
  // BUG: missing refcount++ (dropped fetch_add on the copy ctor).
}

void drop(int s)
{
  if (!slot_live[s])
    return;
  slot_live[s] = 0;
  unsigned prev = refcount--;
  assert(prev >= 1);
  if (prev == 1)
  {
    deleted++;
    assert(deleted == 1);
  }
}

int main(void)
{
  slot_live[0] = 1;
  for (int i = 1; i < N; i++)
    slot_live[i] = 0;

  for (int step = 0; step < K; step++)
  {
    int a = nondet_int(), b = nondet_int();
    __ESBMC_assume(0 <= a && a < N && 0 <= b && b < N);
    if (nondet_bool())
      copy(a, b);
    else
      drop(a);

    unsigned live = 0;
    for (int i = 0; i < N; i++)
      live += slot_live[i];
    assert(live == refcount);
    assert(!(deleted > 0) || refcount == 0);
  }
  return 0;
}
