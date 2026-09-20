/* Both declare_argc_argv call sites match a prefix of `main`, so each of these
   reaches it, and none has the entry point's shape: a two-int signature, a
   bodyless declaration whose first parameter is not an integer, a
   three-parameter form whose third is not a pointer, and one whose first
   parameter is narrower than int (esbmc/esbmc#4715). */
#include <assert.h>

int mainq(double a, char **b);
int mainz(int a, char **b, int c);
int mainc(char a, char **b);

int main_loop(int a, int b)
{
  return a + b;
}

int main(void)
{
  assert(main_loop(1, 2) == 3);
  return 0;
}
