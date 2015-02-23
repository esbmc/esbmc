/* strtok example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] ="- This, a sample string.";
  char * pch;
  std::cout << "Splitting string " << str << " into tokens:" << std::endl;
  pch = strtok (str," ,.-");
  while (pch != NULL)
  {
    std::cout << pch << std::endl;
    pch = strtok (NULL, " ,.-");
  }
  return 0;
}
