/* strcspn example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] = "fcba73";
  char keys[] = "1234567890";
  int i;
  i = strcspn (str,keys);
  std::cout << "The first number in str is at position" << i+1 << std::endl;
  return 0;
}
