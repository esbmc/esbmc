int nondet_int(void);

int main(void)
{
  int a[2][2];
  int(*p)[2] = a;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  /* Coverage, not a gate: pre-PR master proves this too. Reaching the row
     through the array symbol rather than its propagated value routes the read
     through decompose_stores(), so what this pins is that walking a row's own
     chain preserves the verdict. nested_array_row_memcpy is the arm that bites
     when a store is dropped. */
  __ESBMC_assert(p[1][i] == 5 || p[1][i] == 4, "no store the row carries is lost");
  return 0;
}
