#include <cassert>
#include <stdexcept>
#include <deque>
int main()
{
  std::deque<int> d(2, 0);
  int caught = 0;
  try
  {
    d.at(5);
  }
  catch (const std::out_of_range &)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
