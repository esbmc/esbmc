#include <cassert>
#include <array>

int main()
{
  std::array<int, 5> arr1;
  std::array<int, 5> arr2;

  // fill()
  arr1.fill(10);
  arr2.fill(20);
  for (std::size_t i = 0; i < arr1.size(); ++i)
  {
    assert(arr1[i] == 10);
    assert(arr2[i] == 20);
  }

  // operator== and operator!=
  assert(arr1 != arr2);
  arr2.fill(10);
  assert(arr1 == arr2);

  // front() and back()
  assert(arr1.front() == 10);
  assert(arr1.back() == 10);

  // at()
  arr1.at(0) = 100;
  assert(arr1.at(0) == 100);

  int sum = 0;
  for (auto it = arr1.begin(); it != arr1.end(); ++it)
  {
    sum += *it;
  }
  assert(sum == (100 + 10 * 4));

  // get()
  assert(std::get<0>(arr1) == 100);
  assert(std::get<1>(arr1) == 10);

  // swap()
  std::swap(arr1, arr2);
  assert(arr1[0] == 10);
  assert(arr2[0] == 100);

  return 0;
}
