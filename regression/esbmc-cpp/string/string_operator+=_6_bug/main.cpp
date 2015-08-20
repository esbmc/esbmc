// string assigning
//TEST FAILS
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1, str2, str3;
  str1 = string("x");
  str2 = "f(" + str1 + ") ";
  str3 =  string("=") + string(" ");
  assert(str3 != "= x");
  str3 += str1;
  str2 += str3;
  assert(str2 != "f(x) = x");

  cout << str2  << endl;
  return 0;
}
