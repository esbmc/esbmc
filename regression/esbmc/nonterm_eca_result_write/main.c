// The call result lands in `a`, which calculate_output's guards read: with
// input 1 the first pass sets a = -2 and the second exits, so it terminates.
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
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1)
      return -2;
    a = calculate_output(input);
  }
}
