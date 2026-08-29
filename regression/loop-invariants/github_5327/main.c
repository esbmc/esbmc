/* The two assumes contradict, so every auto-generated overflow check below
 * them is discharged on a dead path. That is the intended safe outcome -- the
 * overflow genuinely cannot happen -- so the vacuity probe must not downgrade
 * the verdict on account of it. */
extern int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0);
  __ESBMC_assume(x < 0);

  int i = 0;
  __ESBMC_loop_invariant(i <= 3);
  while (i < 3)
  {
    i = i + 1;
  }

  int y = x + 1;
  return y;
}
