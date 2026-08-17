/* A clause names its targets in the pre-state: `__ESBMC_assigns(buf[head])`
 * grants the element `head` denoted on entry. This body writes exactly that,
 * so it must verify -- and used to be rejected, because the index was read
 * back after the body, by which point `head` had moved on. */
int buf[1000];
int head;

void push(int v)
{
  __ESBMC_requires(head >= 0 && head < 999);
  __ESBMC_assigns(buf[head], head);
  __ESBMC_ensures(1);
  buf[head] = v;
  head = head + 1;
}

int main()
{
  return 0;
}
