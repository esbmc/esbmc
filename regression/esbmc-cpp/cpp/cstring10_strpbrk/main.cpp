/* strpbrk example */
#include <iostream>
#include <cstring>

int main ()
{
  char str[] = "This is a sample string";
  char key[] = "aeiou";
  char * pch;
  std::cout << "Vowels in ' "<< str << " ': " << std::endl;
  pch = strpbrk (str, key);
  while (pch != NULL)
  {
    std::cout << *pch << std::endl;
    pch = strpbrk (pch+1,key);
  }
  std::cout << "\n";
  return 0;
}
