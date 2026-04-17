// Test: unique_ptr<T[]> array specialization with operator[]
#include <cassert>
#include <memory>

int main()
{
  auto arr = std::make_unique<int[]>(3);
  arr[0] = 10;
  arr[1] = 20;
  arr[2] = 30;

  assert(arr[0] == 10);
  assert(arr[1] == 20);
  assert(arr[2] == 30);

  // operator bool on array unique_ptr
  assert(arr);

  // release and manual delete[]
  int *raw = arr.release();
  assert(!arr);
  assert(raw[0] == 10);
  delete[] raw;

  return 0;
}
