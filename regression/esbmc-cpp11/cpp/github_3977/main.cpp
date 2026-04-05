#include <cassert>
int main()
{
  int sum = 0;
  int init[] = {0, 1, 2, 3, 4};
  for (int val : init)
    sum += val;
  assert(sum == 10);
}
