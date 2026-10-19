#include <cassert>
#include <stdexcept>
#include <array>
int main()
{
  std::array<int, 2> a{1, 2};
  int caught = 0;
  try
  {
    int x = a.at(5); (void)x;
  }
  catch (const std::out_of_range &)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
