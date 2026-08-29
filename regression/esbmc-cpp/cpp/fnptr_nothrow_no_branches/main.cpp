#include <iostream>

// No function that may throw has its address taken, so the virtual dispatch
// inside <ostream> must not make main possibly-throwing: a program that only
// writes to cout carries no exception-propagation branch.
int main()
{
  int x = 1;
  std::cout << x;
  return 0;
}
