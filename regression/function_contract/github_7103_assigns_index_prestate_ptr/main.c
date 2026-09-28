/* A clause names its targets in the pre-state: `__ESBMC_assigns(buf[head])`
 * grants the element `head` denoted on entry. This body writes exactly that,
 * so it must verify. The pointer-parameter path read the index back after the
 * body, by which point `head` had moved on -- #7103 for a global array. */
int head;

void push(int *buf, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, 8 * sizeof(int)));
  __ESBMC_requires(head >= 0 && head < 7);
  __ESBMC_assigns(buf[head], head);
  __ESBMC_ensures(1);
  buf[head] = v;
  head = head + 1;
}

int main()
{
  return 0;
}
