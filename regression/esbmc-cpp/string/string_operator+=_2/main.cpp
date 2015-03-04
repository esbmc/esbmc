// string assigning
//Case test operator
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1, str2, str3;
  str1 = "Test string: ";   // c-string
  str2 = 'x';               // single character
  str1 += str2;
  str3 = ", y, " + 'z';
  str1 += str3;

  assert(str1 == "Test string: x, y, z"); //added
  cout << str1  << endl;
  return 0;
}
