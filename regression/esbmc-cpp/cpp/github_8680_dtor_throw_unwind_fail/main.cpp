#include <cassert>
struct T
{
  ~T() noexcept(false) { throw 2; }
};
int main()
{
  int caught = 0;
  try
  {
    T t;
    throw 1;
  }
  catch (int k)
  {
    caught = k;
  }
  assert(caught != 0);
  return 0;
}
