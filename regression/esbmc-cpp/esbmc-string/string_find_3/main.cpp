// string::find
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("To needles i ha need.");
  string str2 ("needle");
  size_t found;

  // different member versions of find in the same order as above:
  found=str.find(str2);
  assert(int(found)==3);
  if (found!=string::npos)
    cout << "first 'needle' found at: " << int(found) << endl;

  found=str.find("needles are small",found+1,3);
  assert(int(found)==16);
  if (found!=string::npos)
    cout << "second 'needle' found at: " << int(found) << endl;

  found=str.find("ha");
  assert(int(found)==13);
  if (found!=string::npos)
    cout << "'haystack' also found at: " << int(found) << endl;

  return 0;
}
