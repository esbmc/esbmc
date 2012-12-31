/* memmove example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] = "memmove can be very useful......";
  memmove (str+20,str+15,11);
  std::cout << str << std::endl;
  return 0;
}
