/* strcpy example */
#include <cstring>
#include <iostream>
#include <cassert>

int main ()
{
  char str1[]="Sample string";
  char str2[40];
  char str3[40];
  strcpy (str2,str1);
  strcpy (str3,"copy successful");
  std::cout << "\nstr1: " << str1 << "\nstr2: " << str2 << "\nstr3: " << str3 << std::endl;
  assert(strcmp(str1,str2)==0);
  assert(strcmp(str3,"copy successful")==0);
  return 0;
}
