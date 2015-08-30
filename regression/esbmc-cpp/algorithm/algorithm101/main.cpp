// partial_sort example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }
//1 2 3 5 4
int main () {
  int myints[] = {5,4,3,2,1};
  vector<int> myvector (myints, myints+5);
  vector<int>::iterator it;
  cout << endl;
  // using default comparison (operator <):
  partial_sort (myvector.begin(), myvector.begin()+3, myvector.end());
  assert(myvector[0] == 1);
  assert(myvector[1] == 2);
  assert(myvector[2] == 3);
  assert(myvector[3] != 5);
  assert(myvector[4] == 4);
 
    /*
  // using function as comp
  cout << endl;
  myvector.assign(myints,myints+5);
  partial_sort (myvector.begin(), myvector.begin()+3, myvector.end(),myfunction);

  assert(myvector[0] == 1);
  assert(myvector[1] != 2);
  assert(myvector[2] == 3);
  assert(myvector[3] == 5);
  assert(myvector[4] == 4);
*/
  // print out content:
  cout << endl;
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
