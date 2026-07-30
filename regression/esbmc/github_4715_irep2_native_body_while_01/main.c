// Exercises the --irep2-native-body dispatcher's code_while2t handling on the
// C frontend (W1-loc spike Phase C, esbmc/esbmc#4715). A plain C assignment
// *statement* (`s = s + i;`) is represented as a code_expression2t wrapping a
// side-effecting assign expression, not a bare code_assign2t (unlike Python,
// whose assignment is a genuine statement) - the code_expression2t kind
// correctly refuses a side-effecting operand, so this loop body falls back
// to goto_convert_rec whole, exactly as a plain-statement while loop already
// did before this kind existed. This pins that the fallback still produces
// byte-identical GOTO and does not corrupt the loop's result, nested loops
// included, alongside the Python test that exercises the native path
// directly (github_4715_irep2_native_body_while_01, whose assignments are
// genuine code_assign2t statements).
#include <assert.h>

int sum_to(int n)
{
  int s = 0;
  int i = 0;
  while (i < n)
  {
    s = s + i;
    i = i + 1;
  }
  return s;
}

int nested(int n)
{
  int i = 0;
  int total = 0;
  while (i < n)
  {
    int j = 0;
    while (j < n)
    {
      total = total + 1;
      j = j + 1;
    }
    i = i + 1;
  }
  return total;
}

int main(void)
{
  assert(sum_to(5) == 10);
  assert(nested(3) == 9);
  return 0;
}
