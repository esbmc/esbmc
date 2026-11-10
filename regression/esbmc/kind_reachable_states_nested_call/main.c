// kind_reachable_states, with the updater storing what nested calls return.
// Resolving those calls while the outer call ran left its call site dangling.
#include <assert.h>
extern int __VERIFIER_nondet_int(void);
int a = 0;
int b = 0;
int c = 0;
int one(void)
{
  return 1;
}
int zero(void)
{
  return one() - 1;
}
int step(int in)
{
  if (a == 1 && b == 1)
  {
    c = c + 1;
    if (c >= 20)
      assert(0);
    return 0;
  }
  if (a == 0 && b == 0 && in == 1)
  {
    a = one();
    return 1;
  }
  if (a == 1 && b == 0 && in == 2)
  {
    a = zero();
    b = one();
    return 2;
  }
  if (a == 0 && b == 1 && in == 1)
  {
    b = zero();
    return 3;
  }
  return -2;
}
int main(void)
{
  int out = 0;
  while (1)
  {
    int in = __VERIFIER_nondet_int();
    if (in != 1 && in != 2)
      return out;
    out = step(in);
  }
}
