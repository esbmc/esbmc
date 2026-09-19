// Issue #5224: the array-element write through a pointer happens inside a
// helper called from the loop, so it is discovered via the function-summary
// path, not the direct assignment path. The helper's pointer resolves to a,
// which the inductive step havocs. This program is safe.
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
