#include <cassert>
int main()
{
  int init[] = {1, -2, 3, -4, 5};
  int positives = 0;
  for (int val : init)
    if (val > 0)
      positives++;
  assert(positives == 3);
}
