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


  str.append("here: ");                   // "here: "
  assert(str == "here: ");
  cout << str << endl;
  return 0;
}
