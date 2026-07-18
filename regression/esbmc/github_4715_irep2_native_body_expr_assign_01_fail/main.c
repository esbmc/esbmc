// _fail sibling of github_4715_irep2_native_body_expr_assign_01: pins that a
// genuine violation reached through natively-converted assignment statements is
// still reported as VERIFICATION FAILED, not silently dropped.
#include <assert.h>

int g;

int f(int a, int b)
{
  int x = a;
  x = x + b;
  g = x;
  while (x < 10)
  {
    x = x + 1;
  }
  if (x > 0)
  {
    g = x;
  }
  else
  {
    g = 0;
  }
  return g;
}

int main(void)
{
  assert(f(1, 2) == 11);
  return 0;
}
