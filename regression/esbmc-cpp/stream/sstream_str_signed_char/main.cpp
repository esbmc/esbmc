// stringstream::operator<<(signed char) must emit the character itself,
// not its decimal value. Without an explicit overload, signed char would
// promote to int and print as a number.
#include <sstream>
#include <cassert>
using namespace std;

int main()
{
  stringstream oss;
  signed char val = 'Y';
  oss << val;
  assert(oss.str() == "Y");
  return 0;
}
