/* strncat example */
#include <iostream>
#include <cstring>

int main ()
{
  char str1[20] = "To be ";
  char str2[20] = "or not to be";
  char *str;
  str = strncat (str1, str2, 6);
  std::cout << str << std::endl;
  return 0;
}
