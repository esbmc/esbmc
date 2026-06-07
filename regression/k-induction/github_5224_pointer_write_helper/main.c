// Issue #5224: the array-element write through a pointer happens inside a
// helper called from the loop, so the un-havocable write is discovered via
// the function-summary path (collect_lhs_symbols), not the direct
// assignment path. The inductive step must still be disabled. This program
// is safe and is proven by the forward condition once the IS is off.
extern unsigned char nondet_uchar(void);

static void setelem(unsigned char (*dest)[8], int i)
{
  (*dest)[i] = nondet_uchar() & 1u; // array element written through a pointer
}

int main(void)
{
  unsigned char a[8];

  for (int i = 0; i < 8; i++)
    setelem(&a, i);

  __ESBMC_assert(a[3] <= 1, "each element was set to 0 or 1");
  return 0;
}
