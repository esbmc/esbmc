// string assigning
//TEST FAILS
//Case test operator
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1;
  str1 = string("Test string");
  str1[2] = 'X';
  assert(str1 != "TeXt string");

  return 0;
}
