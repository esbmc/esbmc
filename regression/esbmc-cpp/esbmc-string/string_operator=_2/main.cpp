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
  str2 = "x";               // single character
  str3 = str1 + str2;       // string

  str3 += ", y, ";
  str3 += "z";

  assert(str3 == "Test string: x, y, z"); //added
  cout << str3  << endl;
  return 0;
}
