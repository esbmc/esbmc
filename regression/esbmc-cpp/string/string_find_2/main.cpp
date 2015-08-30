// string::find
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("Ttwo need hneed.");
  string str2 ("need");
  size_t found;

  // different member versions of find in the same order as above:
  found=str.find(str2);
  assert(int(found)==5);
  if (found!=string::npos)
    cout << "first 'needle' found at: " << int(found) << endl;

  found=str.find("needles are small",found+1,4);
  assert(int(found)==11);
  if (found!=string::npos)
    cout << "second 'needle' found at: " << int(found) << endl;

  return 0;
}
