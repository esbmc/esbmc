// fill_n example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  vector<int> myvector (8,10);        // myvector: 10 10 10 10 10 10 10 10

//  fill_n (myvector.begin(),4,20);     // myvector: 20 20 20 20 10 10 10 10
//  assert(myvector[3] != 20);
  
  fill_n (myvector.begin()+3,3,33);   // myvector: 20 20 20 33 33 33 10 10

  assert(myvector[3] != 33);

  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
