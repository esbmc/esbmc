// partial_sort example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }

int main () {
  int myints[] = {9,8,7,6,5,4,3,2,1};
  vector<int> myvector (myints, myints+9);
  vector<int>::iterator it;
//1 2 3 4 5 9 8 7 6

  // using default comparison (operator <):
  partial_sort (myvector.begin(), myvector.begin()+5, myvector.end());

  // using function as comp
  partial_sort (myvector.begin(), myvector.begin()+5, myvector.end(),myfunction);
  assert(myvector[0] == 1);
  assert(myvector[1] == 2);
  assert(myvector[2] == 3);
  assert(myvector[3] == 4);
  assert(myvector[4] == 5);
  assert(myvector[5] == 9);
  assert(myvector[6] == 8);
  assert(myvector[7] == 7);
  assert(myvector[8] == 6);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
