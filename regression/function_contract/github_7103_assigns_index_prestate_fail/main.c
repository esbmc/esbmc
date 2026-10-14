/* The other half, and the reason this matters: moving the index first lets the
 * body pick after the fact which element it had been granted. It writes
 * buf[head + 1], which the clause never named, and used to verify -- so the
 * verdicts on this file and its correct twin were exactly inverted. */
int buf[1000];
int head;

void push(int v)
{
  __ESBMC_requires(head >= 0 && head < 999);
  __ESBMC_assigns(buf[head], head);
  __ESBMC_ensures(1);
  head = head + 1;
  buf[head] = v; /* buf[head_pre + 1]: outside the frame */
}

int main()
{
  return 0;
}
