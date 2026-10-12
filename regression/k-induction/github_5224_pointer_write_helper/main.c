// Issue #5224: the array-element write through a pointer happens inside a
// helper called from the loop, so it is discovered via the function-summary
// path, not the direct assignment path. The helper's pointer resolves to a,
// which the inductive step havocs. The loop is unbounded, so neither the base
// case nor the forward condition can prove the program safe: only the
// inductive step can, over the havoced a, since the element just written is 0
// or 1 whatever a held.
extern unsigned char nondet_uchar(void);
extern unsigned nondet_uint(void);
extern int nondet_int(void);

static void setelem(unsigned char (*dest)[8], unsigned i)
{
  (*dest)[i] = nondet_uchar() & 1u; // array element written through a pointer
}

int main(void)
{
  unsigned char a[8] = {0};

  while (nondet_int())
  {
    unsigned i = nondet_uint() % 8;
    setelem(&a, i);
    __ESBMC_assert(a[i] <= 1, "the element just set is 0 or 1");
  }
  return 0;
}
