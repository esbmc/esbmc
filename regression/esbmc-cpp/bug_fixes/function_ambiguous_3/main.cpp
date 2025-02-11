#include <cassert>
// first name space
namespace first_space
{
  void func()
  {
     assert(1);
  }
}
 
// second name space
namespace second_space
{
  void func()
  {
     assert(0);
  }
}
using namespace first_space;
int main ()
{
   // This calls function from first name space.
  func();
  return 0;
}