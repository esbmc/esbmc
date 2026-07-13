// H-A1 — Refcount conservation & single-free (Tier A, P0-mem)
//
// Faithful model of the irep_container<T> intrusive-atomic-refcount
// lifecycle (src/irep2/irep2.h):
//   - copy ctor / copy-assign : refcount.fetch_add(1)      (irep2.h:177,205)
//   - release()               : prev = refcount.fetch_sub(1);
//                               delete pointee iff prev == 1   (irep2.h:406-410)
// A container can only *copy* the node from another live container, so a
// freed node can never be resurrected (matches container semantics and
// removes the spurious "adopt a freed node" counterexample).
//
// Property verified: across any sequence of copy/drop on N container slots
// sharing one node, refcount always equals the number of live slots (I1),
// the node is deleted exactly once, and refcount never underflows.

#include <assert.h>

#define N 3 // container slots sharing the node
#define K 5 // steps: enough to fill all slots (copy) then fully drain (drop)

int nondet_int(void);
_Bool nondet_bool(void);

// The node's refcount is the ground truth; slot 0 is seeded live at
// refcount 1 to model the make_irep that created the node.
unsigned refcount = 1;
int deleted = 0;
_Bool slot_live[N];

// Copy the node from a live source `src` into a dead slot `dst`
// (copy ctor: fetch_add on the live source's node).
void copy(int src, int dst)
{
  if (!slot_live[src] || slot_live[dst])
    return; // source must be live, dest must be free
  slot_live[dst] = 1;
  refcount++; // fetch_add(1)
}

// Drop slot `s` (release(): fetch_sub, delete on 1->0).
void drop(int s)
{
  if (!slot_live[s])
    return;
  slot_live[s] = 0;
  unsigned prev = refcount--; // fetch_sub(1)
  assert(prev >= 1);          // no underflow (I1)
  if (prev == 1)
  {
    deleted++;
    assert(deleted == 1); // exactly one free
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

    // Invariant: live slot count == refcount (I1).
    unsigned live = 0;
    for (int i = 0; i < N; i++)
      live += slot_live[i];
    assert(live == refcount);
    // Once freed, nobody may still point at the node.
    assert(!(deleted > 0) || refcount == 0);
  }
  return 0;
}
