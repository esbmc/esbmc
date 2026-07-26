// Exercises the --irep2-native-body dispatcher's code_dowhile2t handling
// (W1-loc spike Phase C, esbmc/esbmc#4715). The loops below convert natively
// end to end: a plain do/while, one whose body uses break and continue (whose
// targets this kind installs), and a nested pair.
//
// Native conversion is byte-identical to the fallback by construction, so this
// test cannot tell which path ran - it pins the verdict; byte-identity is
// discharged by the flag-on/flag-off --goto-functions-only sweep.
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  int i = 0;
  do
  {
    i = i + 1;
    s = s + i;
  } while (i < n);
  return s;
}

int skip_and_stop(void)
{
  int i = 0;
  int s = 0;
  do
  {
    i = i + 1;
    if (i == 3)
      continue;
    if (i > 6)
      break;
    s = s + i;
  } while (i < 10);
  return s;
}

int nested(int n)
{
  int i = 0;
  int total = 0;
  do
  {
    int j = 0;
    do
    {
      total = total + 1;
      j = j + 1;
    } while (j < n);
    i = i + 1;
  } while (i < n);
  return total;
}

int main(void)
{
  assert(sum_to(5) == 15);
  assert(skip_and_stop() == 1 + 2 + 4 + 5 + 6);
  assert(nested(3) == 9);
  return 0;
}
