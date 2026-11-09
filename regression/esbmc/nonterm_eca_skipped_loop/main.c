// The loop never runs: the jump before it always skips it, so the program
// ends although the loop alone would repeat a state (#10005).
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);
int a = 1;
int calculate_output(int input)
{
  if (a == -2)
    exit(0);
  if (a == 2 && input == 1)
  {
    a = 1;
    return 0;
  }
  return -2;
}
int main(void)
{
  int skip = 1;
  int output = 0;
  if (skip)
    goto end;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1)
      return -2;
    output = calculate_output(input);
  }
end:
  return output;
}
