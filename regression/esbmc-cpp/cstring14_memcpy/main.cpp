/* memcpy example */
#include <iostream>
#include <cstring>

int main ()
{
  char str1[]="Sample string";
  char str2[40];
  char str3[40];
  memcpy (str2,str1,14);
  memcpy (str3,"copy successful",16);
  std::cout << "str1: " << str1 << "\nstr2: " << str2 << "\nstr3: " << str3 << std::endl;
  return 0;
}
