// appending to string
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str;
  string str3="print 10 and then 5 more";
  str.append(str3.begin()+8,str3.end());  // " and then 5 more"
  assert(str == " and then 5 more");
  cout << str << endl;
  return 0;
}
