// Issue #5224: the helper writes a[3] through a pointer only in iteration 20.
// Unhavoced, a[3] would keep its initial 0 in the inductive step, which would
// prove the assertion; with a havoced the base case finds the violation.
extern unsigned char nondet_uchar(void);
extern int nondet_int(void);

static void setelem(unsigned char (*dest)[8], unsigned i)
{
  (*dest)[i] = nondet_uchar() | 1u; // array element written through a pointer
}

int main(void)
{
  unsigned char a[8] = {0};
  unsigned n = 0;

  while (nondet_int())
  {
    if (n == 20)
      setelem(&a, 3);
    n++;
  }
  __ESBMC_assert(a[3] == 0, "a[3] is written in iteration 20");
  return 0;
}
