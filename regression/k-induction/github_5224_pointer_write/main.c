// Issue #5224 / #5230: a loop that writes array elements through a pointer
// (here a pointer-to-array, the same shape SV-COMP's Intel-TDX-Module
// harnesses use via __VERIFIER_nondet_array_1D_*) cannot be havoc'd as a
// named symbol by the k-induction inductive step. Before #5224 the inductive
// step proved this unsafe program "true". Phase 2 (#5230) resolves the
// written pointer `dest` against the value-set fixpoint to the named object
// `a` and havocs `a` as a whole symbol, so the inductive step stays enabled
// and sound: it cannot prove the property (a[7] is nondet) and the base case
// finds the genuinely reachable violation. The pointer-write warning is NOT
// emitted because the write is resolved, not gated.
extern unsigned char nondet_uchar(void);

int main(void)
{
  unsigned char a[8];
  unsigned char (*dest)[8] = &a;

  for (int i = 0; i < 8; i++)
    (*dest)[i] = nondet_uchar(); // array element written through a pointer

  __ESBMC_assert(a[7] != 0x2A, "a[7] is nondet, so 0x2A is reachable");
  return 0;
}
