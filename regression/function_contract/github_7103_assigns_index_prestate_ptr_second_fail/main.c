/* Moving `tail` first writes buf[tail + 1], which the clause never granted --
 * `head` is unconstrained against it, so it is not excused by the other named
 * index either. Must be caught even though the offending index is the second
 * one the clause names. */
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
