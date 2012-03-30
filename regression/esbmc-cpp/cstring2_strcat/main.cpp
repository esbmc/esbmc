/* strcat example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[80] = "these ";
  strcat (str,"strings ");
  strcat (str,"are ");
  strcat (str,"concatenated.");
  std::cout << str << std::endl;
  return 0;
}
