/* The moving index is the second one the clause names, so this is caught only
 * if every index in the group is captured on entry, not just the first. Its
 * in-frame counterpart is not a guard: that direction verifies whether or not
 * the index was captured, so only the out-of-frame write discriminates. */
int head;
int tail;

void push_tail_bad(int *buf, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, 8 * sizeof(int)));
  __ESBMC_requires(head >= 0 && head < 7);
  __ESBMC_requires(tail >= 0 && tail < 7);
  __ESBMC_assigns(buf[head], buf[tail], tail);
  __ESBMC_ensures(1);
  tail = tail + 1;
  buf[tail] = v;
}

int main()
{
  return 0;
}
