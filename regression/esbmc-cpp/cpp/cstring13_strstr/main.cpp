/* strstr example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] ="This is a simple string";
  char * pch;
  pch = strstr (str,"simple");
  std::cout << str << "\nresult: " << pch << std::endl;
  return 0;
}
