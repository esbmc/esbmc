/*
 * For Issue 991, we get a reference to the class type rather than copying it.
 * This TC aims to confirm that ESBMC backend is happy with this approach,
 * and shall not report any error/assertion failure.
 */
#include <cassert>

class string
{
public:
  class iterator
  {
  public:
    int data;
  };

  iterator itr;
};

int main()
{
  string str;
  assert(1);
  return 0;
}
