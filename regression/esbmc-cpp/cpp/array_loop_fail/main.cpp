#include <cassert>

struct Example
{
  int arr[3];

  Example() : arr{1, 2, 3}
  {
  }
};

int main()
{
  Example original;
  Example copy = original;

  assert(copy.arr[0] == 1);
  assert(copy.arr[1] == 2);
  assert(copy.arr[2] == 0); // fail

  return 0;
}
