// next_permutation
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

int main () {
  int myints[] = {1,2,3};

  cout << "The 3! possible permutations with 3 elements:\n";

  sort (myints,myints+3);
  next_permutation (myints,myints+3);
  assert(myints[2] == 3);
  
  return 0;
}
