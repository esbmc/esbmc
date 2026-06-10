// Issue #5224: a loop that writes array elements through a pointer (here a
// pointer-to-array, the same shape SV-COMP's Intel-TDX-Module harnesses use
// via __VERIFIER_nondet_array_1D_*) cannot be soundly havoc'd by the
// k-induction inductive step, which havocs only named symbols. Before the
// fix the inductive step proved this unsafe program "true"; the fix disables
// the inductive step for such loops, so the base case finds the genuinely
// reachable violation instead.
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
