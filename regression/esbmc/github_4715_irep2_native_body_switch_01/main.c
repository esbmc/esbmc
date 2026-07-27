// Exercises the --irep2-native-body dispatcher's code_switch2t /
// code_switch_case2t handling (W1-loc spike Phase C, esbmc/esbmc#4715). Each of
// these converts natively end to end: consecutive case labels sharing an arm,
// fallthrough, a switch with no default, a default that is not last, a nested
// pair, a switch inside a loop mixing break and continue, an arm introducing a
// scoped declaration, and an empty body.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot tell which path ran - it pins the verdict; byte-identity is
// discharged by the flag-on/flag-off --goto-functions-only sweep.
#include <assert.h>

int shared_arm(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
    x = 10;
    break;
  case 2:
  case 3:
    x = 20;
    break;
  default:
    x = 99;
  }
  return x;
}

int fallthrough(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
    x = 1;
  case 2:
    x = x + 2;
    break;
  default:
    x = 9;
  }
  return x;
}

int no_default(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
    x = 1;
    break;
  case 2:
    x = 2;
    break;
  }
  return x;
}

int default_in_the_middle(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
    x = 1;
    break;
  default:
    x = 7;
    break;
  case 2:
    x = 2;
  }
  return x;
}

int nested(int a, int b)
{
  int x = 0;
  switch (a)
  {
  case 1:
    switch (b)
    {
    case 1:
      x = 11;
      break;
    default:
      x = 19;
    }
    break;
  default:
    x = 90;
  }
  return x;
}

int in_loop(int n)
{
  int x = 0;
  for (int i = 0; i < n; i++)
  {
    switch (i)
    {
    case 0:
      continue;
    case 1:
      x = x + 1;
      break;
    default:
      x = x + 2;
    }
    x = x + 100;
  }
  return x;
}

int scoped_decl(int a)
{
  int x = 0;
  switch (a)
  {
  case 1:
  {
    int y = 5;
    x = y;
    break;
  }
  default:
    x = 3;
  }
  return x;
}

int empty_body(int a)
{
  int x = 4;
  switch (a)
  {
  }
  return x;
}

int main(void)
{
  assert(shared_arm(1) == 10);
  assert(shared_arm(3) == 20);
  assert(shared_arm(7) == 99);
  assert(fallthrough(1) == 3);
  assert(fallthrough(2) == 2);
  assert(fallthrough(5) == 9);
  assert(no_default(1) == 1);
  assert(no_default(5) == 0);
  assert(default_in_the_middle(2) == 2);
  assert(default_in_the_middle(5) == 7);
  assert(nested(1, 1) == 11);
  assert(nested(1, 4) == 19);
  assert(nested(4, 1) == 90);
  assert(in_loop(4) == 1 + 100 + 2 + 100 + 2 + 100);
  assert(scoped_decl(1) == 5);
  assert(empty_body(0) == 4);
  return 0;
}
