/* strlen example */
#include <iostream>
#include <cstring>
#include <cassert>

int main ()
{
  char szInput[256] = "test of the string";
  std::cout << "The sentence entered is " << (unsigned)strlen(szInput) << " characters long" << std::endl;
  assert((unsigned)strlen(szInput)==18);

  return 0;
}
