/* The invariant spells its operand behind a `?:`, so the frontend computes it
 * under a jump between count_to's call and the marker. That value cannot be
 * replayed after the havoc in order, so the invariant is refused rather than
 * assumed from its stale pre-loop value. */
static unsigned int count_to(unsigned int n)
{
  unsigned int i = 0;
  unsigned int s = 0;
  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }
  return s;
}

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant((i <= 4) ? (count_to(4) == 4) : 0);
  while (i < 4)
    i++;

  __ESBMC_assert(i == 4, "end");
  return 0;
}
