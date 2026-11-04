// The call result lands in a local the guards never read, so with a == 1 and
// input 1 no branch fires and the state repeats. The terminating loop before
// it must not take the cycle waypoint.
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
  int output = 0;
  for (int i = 0; i < 3; i++)
    output = i;
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1)
      return -2;
    output = calculate_output(input);
  }
}
