/* The loop that must be left alone sits two calls below the marker, so reaching
 * it is the call-graph closure's worklist and nothing else: protecting only the
 * callees named in the marker's own block stops at outer(), and count_to's loop
 * is then cut. The marker would read a havoc-abstracted return -- the user's
 * invariant evaluated on a value the loop never produced. */
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

static unsigned int mid(unsigned int n)
{
  return count_to(n);
}

static unsigned int outer(unsigned int n)
{
  return mid(n);
}

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(outer(4) == 4);
  while (i < 4)
    i++;

  __ESBMC_assert(i == 4, "end");
  return 0;
}
