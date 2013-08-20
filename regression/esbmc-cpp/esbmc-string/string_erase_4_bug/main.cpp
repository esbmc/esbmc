// string::erase
//TEST FAILS
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("This is an example phrase.");
  string::iterator it;

  // erase used in the same order as described above:
  str.erase (10,8);
  assert(str != "This is an phrase.");
  cout << str << endl;        // "This is an phrase."

  it=str.begin()+9;
  str.erase (it);
  assert(str != "This is a phrase.");
  cout << str << endl;        // "This is a phrase."

  return 0;
}
