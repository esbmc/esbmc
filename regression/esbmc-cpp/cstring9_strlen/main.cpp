/* strlen example */
#include <iostream>
#include <cstring>

int main ()
{
  char szInput[256] = "test of the string";
  std::cout << "The sentence entered is " << (unsigned)strlen(szInput) << " characters long" << std::endl;
  return 0;
}
