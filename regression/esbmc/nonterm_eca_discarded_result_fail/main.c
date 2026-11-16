// The loop discards calculate_output's result. With a == 1 and input 1 no
// branch fires and the state repeats (#10006).
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
    calculate_output(input);
  }
}
