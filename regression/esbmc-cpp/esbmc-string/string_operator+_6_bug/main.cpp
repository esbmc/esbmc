// string assigning
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str1, str2;
  str1 = 'x';
  str2 = " f(" + str1 + ") " + '=' + ' ' + str1;

  assert(str2 != " f(x) = x");

  cout << str2  << endl;
  return 0;
}
