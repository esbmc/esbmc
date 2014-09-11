// nth_element example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }
int main () {
  vector<int> myvector;
  vector<int>::iterator it;
  // 5 2 3 4 1 6 7 8 9
  // set some values:
  for (int i=1; i<10; i++) myvector.push_back(i);   // 1 2 3 4 5 6 7 8 9

  // using default comparison (operator <):
  /*
  nth_element (myvector.begin(), myvector.begin()+5, myvector.end());
  assert(myvector[5] != 6);
  assert(myvector[6] >= 6);
  assert(myvector[4] <= 6);
 */
  // using function as comp
  nth_element (myvector.begin(), myvector.begin()+5, myvector.end(),myfunction);
  assert(myvector[5] != 6);
  assert(myvector[4] <= 6);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
