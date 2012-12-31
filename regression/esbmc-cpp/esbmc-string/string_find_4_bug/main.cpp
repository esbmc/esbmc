//TEST FAILS
// string::find
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("There are two needles ");
  string str2 ("needle");
  size_t found;

  // different member versions of find in the same order as above:
  found=str.find(str2);
  assert(int(found)!=14);
  if (found!=string::npos)
    cout << "first 'needle' found at: " << int(found) << endl;


  return 0;
}
