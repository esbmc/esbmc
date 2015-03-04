// partial_sort_copy example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

//Myvector: 1 2 3 4 5 
//myints: 8 7 6 5 4 3 2 1 
//Myvector: 1 2 3 4 5 
//myints: 8 7 6 5 4 3 2 1 

bool myfunction (int i,int j) { return (i<j); }

int main () {
  int myints[] = {8,7,6,5,4,3,2,1};
  vector<int> myvector (5);
  vector<int>::iterator it;

  // using default comparison (operator <):
  /*
  partial_sort_copy (myints, myints+8, myvector.begin(), myvector.end());
  assert(myvector[0] == 1);
  assert(myvector[1] == 2);
  assert(myvector[2] == 3);
  assert(myvector[3] == 4);
  assert(myvector[4] == 5);
  */
  // using function as comp
  
  partial_sort_copy (myints, myints+8, myvector.begin(), myvector.end(), myfunction);
  assert(myvector[0] == 1);
  assert(myvector[1] == 2);
  assert(myvector[2] != 3);
  assert(myvector[3] == 4);
  assert(myvector[4] == 5);

  cout << endl;

  return 0;
}
