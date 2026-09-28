// _fail sibling of github_4715_irep2_native_body_goto_01: pins that a genuine
// violation reached through natively-converted goto/label statements is still
// reported as VERIFICATION FAILED, not silently dropped.
#include <assert.h>

int forward(int a)
{
  int x = a;
  if (x < 0)
    goto neg;
  x += 10;
  goto done;
neg:
  x = -x;
done:
  return x;
}

int scope_jump(int a)
{
  int x = a;
  {
    int y = x + 1;
    if (y > 3)
      goto out;
    x = y;
  }
out:
  return x;
}

int backward(int a)
{
  int i = 0;
top:
  i += 1;
  if (i < a)
    goto top;
  return i;
}

int main(void)
{
  assert(forward(1) == 11);
  assert(forward(-2) == 2);
  assert(scope_jump(5) == 5);
  assert(scope_jump(1) == 3);
  assert(backward(3) == 3);
  return 0;
}
