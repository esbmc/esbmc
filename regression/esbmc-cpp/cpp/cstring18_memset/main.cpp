/* memset example */
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
#include <cstring>

int main ()
{
  char str[] = "almost every programmer should know memset!";
  memset (str,'-',6);
  cout << str << endl;
  return 0;
}
