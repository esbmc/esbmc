int nondet_int(void);

int main(void)
{
  int a[2][2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  /* Coverage, not a gate: pre-PR master proves this too. The nondet row index
     sends the read through lower_flattened_row_select(), so what this pins is
     that pushing a select into a row's store chain leaves both writes
     readable. nested_array_plane_memcpy is the arm that bites when one is
     lost. */
  __ESBMC_assert(a[1][i] == 5 || a[1][i] == 4, "both row stores reach the solver");
  return 0;
}
