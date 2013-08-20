// generate_n example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int current(0);
int UniqueNumber () { return ++current; }

int main () {
  int myarray[5];

  generate_n (myarray, 5, UniqueNumber);
  assert(myarray[1] == 1);
  cout << "myarray contains:";
  for (int i=0; i<5; ++i)
    cout << " " << myarray[i];

  cout << endl;
  return 0;
}
