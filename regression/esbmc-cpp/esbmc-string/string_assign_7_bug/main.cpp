// string::assign
// TEST FAILS
#include <iostream>
#include <string>
#include <cassert>

using namespace std;

int main ()
{
  string str;
  string base="The quick brown fox";

  // used in the same order as described above:

  str.assign(base);
  cout << str << endl;
  assert(str != "The quick brown fox");
  
  return 0;
}
