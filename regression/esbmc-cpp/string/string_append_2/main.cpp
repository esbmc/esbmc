// appending to string
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string str;
  string str2="Writing ";
  string str3="print 10 and then 5 more";

  // used in the same order as described above:
  
  str.append(str3,6,3);                   // "10 "
  assert(str == "10 ");

  cout << str << endl;
  return 0;
}
