// generate_n example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int current(0);
int UniqueNumber () { return ++current; }

int main () {
  int myarray[9];

  generate_n (myarray, 9, UniqueNumber);

//  cout << "myarray contains:";
  for (int i=0; i<9; ++i) assert(myarray[i] == i+1);
//    cout << " " << myarray[i];

  cout << endl;
  return 0;
}
