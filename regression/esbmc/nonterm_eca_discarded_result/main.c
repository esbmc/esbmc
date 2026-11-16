// The loop discards calculate_output's result. With input 1 the first pass
// sets a = 1 and the second exits, so it terminates (#10006).
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);
int a = 0;
int calculate_output(int input)
{
  if (a == 0 && input == 1)
  {
    a = 1;
    return 1;
  }
  exit(0);
}
int main(void)
{
  while (1)
  {
    int input;
    input = __VERIFIER_nondet_int();
    if (input != 1 && input != 2)
      return -2;
    calculate_output(input);
  }
}
