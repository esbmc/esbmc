// string::size
#include <iostream>
#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str ("Test string");
  int i = (int) str.size();
  assert(i == 11);
  cout << "The size of str is " << str.size() << " characters.\n";
  return 0;
}
