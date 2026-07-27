// Exercises the --irep2-native-body dispatcher's code_for2t handling (W1-loc
// spike Phase C, esbmc/esbmc#4715). Each of these converts natively end to
// end: a for with a declaration in the init, a nested pair, one whose body
// uses break and continue, and one with the iteration statement omitted
// (which still emits the SKIP the continue target points at).
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot tell which path ran - it pins the verdict; byte-identity is
// discharged by the flag-on/flag-off --goto-functions-only sweep.
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  for (int i = 0; i < n; i = i + 1)
    s = s + i;
  return s;
}

int nested(int n)
{
  int t = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      t = t + 1;
  return t;
}

int skip_and_stop(int n)
{
  int s = 0;
  for (int i = 0; i < n; i++)
  {
    if (i == 2)
      continue;
    if (i > 5)
      break;
    s = s + i;
  }
  return s;
}

int no_iter(int n)
{
  int i = 0;
  for (; i < n;)
    i = i + 1;
  return i;
}

int main(void)
{
  assert(sum_to(5) == 10);
  assert(nested(3) == 9);
  assert(skip_and_stop(9) == 0 + 1 + 3 + 4 + 5);
  assert(no_iter(4) == 4);
  return 0;
}
