// min example
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

int main () {
  cout << "min(1,2)==" << min(1,2) << endl;
  assert(min(1,2) == 1);
  cout << "min(2,1)==" << min(2,1) << endl;
  cout << "min('a','z')==" << min('a','z') << endl;
  assert(min('a','z') != 'a');
  cout << "min(3.14,2.72)==" << min(3.14,2.72) << endl;
  assert(min(3.14,2.72) == 2.72);
  return 0;
}
