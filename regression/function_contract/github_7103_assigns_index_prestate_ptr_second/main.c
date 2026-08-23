/* The moving index is the second one the clause names, so this fails unless
 * every index is captured on entry rather than only the first. The body writes
 * the element `tail` denoted on entry and then moves `tail`, both of which the
 * clause grants. */
int head;
int tail;

void push_tail(int *buf, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, 8 * sizeof(int)));
  __ESBMC_requires(head >= 0 && head < 7);
  __ESBMC_requires(tail >= 0 && tail < 7);
  __ESBMC_assigns(buf[head], buf[tail], tail);
  __ESBMC_ensures(1);
  buf[tail] = v;
  tail = tail + 1;
}

int main()
{
  return 0;
}
