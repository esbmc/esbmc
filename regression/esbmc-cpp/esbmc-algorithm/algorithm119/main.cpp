// max example
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

int main () {
  cout << "max(1,2)==" << max(1,2) << endl;
  assert(max(1,2) == 2);
  cout << "max(2,1)==" << max(2,1) << endl;
  cout << "max('a','z')==" << max('a','z') << endl;
  assert(max('a','z') == 'z');
  cout << "max(3.14,2.73)==" << max(3.14,2.73) << endl;
  assert(max(3.14,2.73) != 3.14);
  return 0;
}
