// _fail sibling of github_4715_irep2_native_body_expr_sideeffect_01: pins that a
// genuine violation reached through natively-lowered side-effecting expression
// statements is still reported as VERIFICATION FAILED, not silently dropped.
#include <assert.h>

int g;

void sink(int v)
{
  g = v;
}

int f(int a)
{
  int x = a;
  x += 3;
  x -= 1;
  x *= 2;
  ++x;
  x++;
  --x;
  x--;
  sink(x);
  return x;
}

int main(void)
{
  assert(f(1) == 7);
  assert(g == 6);
  return 0;
}
