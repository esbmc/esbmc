#include <typeinfo>
#include <assert.h>
#include <string>

int main(void)
{
  int i;
  int * pi;

  std::string x1 = 
    typeid(int).name();

  std::string x2 = 
    typeid(i).name();

  std::string x3 = 
    typeid(pi).name();

  std::string x4 = 
    typeid(*pi).name();

  assert(x1 == "i");
  assert(x2 == "i");
  assert(x3 == "Pi");
  assert(x4 == "i");

  return 0;
}
