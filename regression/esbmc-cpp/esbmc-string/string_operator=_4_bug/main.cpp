// string assigning
//Example from C++ Reference, avaliable at http://www.cplusplus.com/reference/string/string/operator=/
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1, str2, str3;
  str1 = "Test sTring: ";   // c-string
  str2 = 'x';               // single character
  str3 = str1 + str2;       // string

  assert(str3 != "Test sTring: x"); 		//added
  cout << str3  << endl;
  return 0;
}
