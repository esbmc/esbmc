// The shape that kept the destructor disabled: a class-typed conditional used
// to destroy every branch temporary, so with ~unique_ptr present this reported
// a false `invalid pointer freed`. It is the aws-sdk-cpp pattern the nullptr_t
// constructor in src/cpp/library/memory exists to support.
#include <memory>
#include <cassert>
#include <cstddef>

int main()
{
  std::size_t n = 2;

  std::unique_ptr<int> p = (n > 0) ? std::unique_ptr<int>(new int(5)) : nullptr;
  assert(*p == 5);

  std::unique_ptr<int[]> arr =
    (n > 0) ? std::unique_ptr<int[]>(new int[n]) : nullptr;
  arr[0] = 5;
  assert(arr[0] == 5);

  return 0;
}
