// Extraction has no input to read, so the operand must come back
// unconstrained. Leaving it at its prior value proved this assertion, which
// real input refutes -- a missed bug, not a spurious alarm.
#include <iostream>
#include <cassert>

int main()
{
  int i = 7;
  double d = 2.5;
  long long q = 9;

  std::cin >> i;
  std::cin >> d;
  std::cin >> q;

  assert(i == 7 && d == 2.5 && q == 9);
  return 0;
}
