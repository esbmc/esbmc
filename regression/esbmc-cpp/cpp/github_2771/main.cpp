// esbmc/esbmc#2771: a range-based for loop over a std::vector used to report
// "Couldn't determine upper bound for range-based for loop". It now iterates
// correctly; check the visited elements sum as expected.
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> numbers = {10, 20, 30, 40};
  int sum = 0;
  for (int num : numbers)
    sum += num;
  assert(sum == 100);
  return 0;
}
