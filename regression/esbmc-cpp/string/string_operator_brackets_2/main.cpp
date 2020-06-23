// string assigning
//Case test operator
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1;
  str1 = string("Test string");
  str1[2] = 'x';
  assert(str1 == "Text string");

  return 0;
}
