// Exercises the --irep2-native-body dispatcher's side-effecting-condition
// slices for code_for2t, code_dowhile2t and code_switch2t (W1-loc spike Phase
// C, esbmc/esbmc#4715). Each kind was first landed for the side-effect-free
// condition only, where the lowering preamble is empty and the back-edge /
// continue target collapses onto the guard; these shapes make the preamble
// non-empty, so the target has to point at its first instruction instead.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot tell which path ran - it pins the verdict; byte-identity is
// discharged by the flag-on/flag-off --goto-functions-only sweep.
#include <assert.h>

int counter;

int has_more(void)
{
  counter = counter + 1;
  return counter < 5;
}

int for_call_cond(void)
{
  int s = 0;
  counter = 0;
  for (; has_more();)
    s = s + 1;
  return s;
}

int for_call_cond_and_iter(void)
{
  int s = 0;
  counter = 0;
  for (int i = 0; has_more(); i++)
    s = s + i;
  return s;
}

int dowhile_post_dec(int t)
{
  int s = 0;
  do
  {
    s = s + 1;
  } while (t--);
  return s;
}

int dowhile_call_cond(void)
{
  int s = 0;
  counter = 0;
  do
  {
    s = s + 1;
  } while (has_more());
  return s;
}

int switch_call_value(void)
{
  int x = 0;
  counter = 3;
  switch (has_more())
  {
  case 0:
    x = 100;
    break;
  case 1:
    x = 200;
    break;
  default:
    x = 300;
  }
  return x;
}

int switch_post_inc(int a)
{
  int x = 0;
  switch (a++)
  {
  case 1:
    x = 1;
    break;
  default:
    x = 9;
  }
  return x + a;
}

int main(void)
{
  // has_more() returns true on its first four calls, then false.
  assert(for_call_cond() == 4);
  assert(for_call_cond_and_iter() == 0 + 1 + 2 + 3);
  assert(dowhile_post_dec(2) == 3);
  assert(dowhile_call_cond() == 5);
  assert(switch_call_value() == 200);
  assert(switch_post_inc(1) == 1 + 2);
  return 0;
}
