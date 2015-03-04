// TEST FAILS
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string str;
  string str2="Writing ";
  string str3="print 10 and then 5 more";

  str.append("dots are cool",5);          // "dots "
  assert(str != "dots ");
  cout << str << endl;
  return 0;
}
