// C11 6.5.7p3 promotes each shift operand on its own, so the _Bool left
// operand becomes int and 6.5.7p4 gives the result that type. Left unpromoted
// the one-bit sort reaches the solver where a bitvector is wanted.
int main(void)
{
  int i = 1, j = 5, nc_B = 3;

  int found = (j > nc_B - 1) << i;

  __ESBMC_assert(found == 2, "a _Bool operand promotes to int before the shift");
  return 0;
}
