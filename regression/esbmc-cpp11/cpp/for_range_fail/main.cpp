#include <cassert>

int main()
{
  int arr[] = {1, 2, 3, 4, 5};

  int index = 1;
  for (int num : arr)
  {
    assert(num != index);
    index++;
  }

  return 0;
}
