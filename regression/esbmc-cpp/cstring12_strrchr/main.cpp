/* strrchr example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] = "This is a sample string";
  char * pch;
  pch=strrchr(str,'s');
  std::cout << "Last occurence of 's' found at " << pch-str+1 << std::endl;
  return 0;
}
