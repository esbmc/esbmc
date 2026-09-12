/* The invariant spells its operand behind a `?:`, so the frontend puts a join
 * label between count_to's call and the marker. The extractor still hoists that
 * call, so the dependency scan has to see it too: it takes every call ahead of
 * a marker in the function rather than re-deriving the extractor's rule. */
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
