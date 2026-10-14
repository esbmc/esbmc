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

/* Models the SV-COMP entry harness of coreutils/comm_3args_ok: an argv-style
 * array of char* is stored into a flat malloc'd byte buffer, and the program
 * exit()s (running the leak check) while the buffer and its elements are still
 * reachable from main's own local `a`. None of it is a leak -- valid-memtrack
 * counts stack-reachable memory as tracked. Regresses issue #5138. */
int main(void)
{
  atexit(cleanup); /* registers a handler so exit() runs the leak check */
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
  }
  terminate(a, n); /* exit() while `a` and every a[i] are still reachable */
  for (int i = 0; i < n; i++)
    free(a[i]); /* keeps `a` live across terminate(): not a leak */
  free(a);
  return 0;
}
