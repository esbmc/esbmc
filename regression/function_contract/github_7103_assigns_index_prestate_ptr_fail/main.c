/* Moving `head` first writes buf[head + 1], an element the clause never
 * granted. Reading the index back after the body excused it and verified the
 * broken frame; the index must be the one denoted on entry. */
int head;

void push_bad(int *buf, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, 8 * sizeof(int)));
  __ESBMC_requires(head >= 0 && head < 7);
  __ESBMC_assigns(buf[head], head);
  __ESBMC_ensures(1);
  head = head + 1;
  buf[head] = v;
}

int main()
{
  return 0;
}
