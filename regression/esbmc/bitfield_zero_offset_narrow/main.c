/*
 * Pin: zero-offset narrow bitfield assignments must be modeled.
 *
 * lshr2t::do_simplify folds lshr(val, 0) -> val, so the bitfield write
 * shape `(rtype)(lshr(val, 0)) & mask` collapses to `(rtype)val & mask`.
 * symex_assign_bitfield's no-lshr fallback recognizes this by checking
 * that the mask has contiguous low bits set. The MSB-first binary check
 * must accept masks like 0x0F (00001111), or zero-offset narrow
 * bitfields would be silently dropped.
 *
 * Codex iteration 29 caught a bug where the predicate was rejecting
 * normal narrow masks like 0x0F.
 */
#include <assert.h>

struct S {
  unsigned int a : 4;  // zero offset, 4 bits — mask 0x0F
  unsigned int b : 4;
};

int main() {
  struct S s = {0};
  s.a = 7;
  // Without the fix, the assignment was silently dropped, leaving s.a = 0.
  assert(s.a == 7);
  return 0;
}
