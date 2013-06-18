// prev_permutation
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

int main () {
  int myints[] = {1,2,3};

  cout << "The 3! possible permutations with 3 elements:\n";

  sort (myints,myints+3);
  reverse (myints,myints+3);
//  assert(myints[0] != 3);
  do {
    cout << myints[0] << " " << myints[1] << " " << myints[2] << endl;
    assert(myints[0] == 1);
  } while ( prev_permutation (myints,myints+3) );
  

  return 0;
}
