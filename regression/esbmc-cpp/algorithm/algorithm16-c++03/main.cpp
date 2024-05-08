// swap_ranges example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  vector<int> first (5,10);        //  first: 10 10 10 10 10
  vector<int> second (5,33);       // second: 33 33 33 33 33
  vector<int>::iterator it;

  swap_ranges(first.begin()+1, first.end()-1, second.begin());

//first contains: 10 33 33 33 10
//second contains: 10 10 10 33 33
  assert(first[0] == 10);
  assert(first[1] == 33);
  assert(first[2] == 33);
  assert(first[3] == 33);
  assert(first[4] == 10);
  
  assert(second[0] == 10);
  assert(second[1] == 10);
  assert(second[2] == 10);
  assert(second[3] == 33);
  assert(second[4] == 33);
  // print out results of swap:
  cout << " first contains:";
  for (it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;

  cout << "\nsecond contains:";
  for (it=second.begin(); it!=second.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
