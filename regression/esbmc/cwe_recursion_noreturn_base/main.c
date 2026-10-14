// A recursive function whose base case terminates the program via exit()
// rather than returning normally. This is *controlled* recursion (it bottoms
// out at n <= 0), so exceeding the unwind bound must stay the unmapped
// "recursion unwinding assertion" and must NOT be relabelled CWE-674. Pins the
// noreturn-terminator handling in recursion_has_no_base_case().
#include <stdlib.h>

int f(int n)
{
  if (n <= 0)
    exit(0);
  return f(n - 1);
}

int main(void)
{
  return f(1000);
}
