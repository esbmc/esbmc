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
  }
  catch (int k)
  {
    caught = k;
  }
  assert(caught == 2);
  return 0;
}
