#include <stdlib.h>

extern char nondet_char(void);

static void cleanup(void)
{
}

static void terminate(char **a, int n)
{
  (void)a;
  (void)n;
  exit(0);
}

/* Soundness companion to github_5138: the element pointers are dropped (a[i]=0)
 * before exit(), so the three malloc(8) blocks are genuinely orphaned and must
 * still be reported as leaked. The content-based reachability chase reads the
 * buffer's *current* bytes, which no longer hold the element addresses, so the
 * leak is not masked. */
int main(void)
{
  atexit(cleanup);
  int n = 3;
  char **a = malloc((n + 1) * 8UL);
  if (!a)
    return 0;
  for (int i = 0; i < n; i++)
  {
    a[i] = malloc(8);
    if (!a[i])
      return 0;
    a[i][0] = nondet_char();
    a[i] = 0; /* drop the only reference -> genuine leak of this block */
  }
  terminate(a, n);
  free(a);
  return 0;
}
