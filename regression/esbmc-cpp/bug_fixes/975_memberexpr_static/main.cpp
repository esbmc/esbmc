#include <cassert>
class istreamX
{
public:
  istreamX()
  {
    _gcount = 50;
  }
  static int _gcount;
  //int _gcount; // all good
};

int istreamX::_gcount = 0;

istreamX& test_function(istreamX& is)
{
  is._gcount = 100;
  return is;
}

int main()
{
  assert(istreamX::_gcount == 0);
  istreamX obj;
  assert(istreamX::_gcount == 50);
  test_function(obj);
  assert(istreamX::_gcount == 100);
  return 0;
}
