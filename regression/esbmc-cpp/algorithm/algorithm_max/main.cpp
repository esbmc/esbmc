// max example
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;

int main () {
  cout << "max(1,2)==" << max(1,2) << endl;
  cout << "max(2,1)==" << max(2,1) << endl;
  cout << "max(3.14,2.73)==" << max(3.14,2.73) << endl;
//  cout << "max('a','z')==" << max('a','z') << endl;

  assert(max(3.14,2.73) == 3.14);
  assert(max(2,3) == 3);
  return 0;
}
