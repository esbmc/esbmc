// mismatch algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool mypredicate (int i, int j) {
  return (i==j);
}

int main () {

  vector<int> myvector;

  for (int i=1; i<6; i++) myvector.push_back (i*10); // myvector: 10 20 30 40 50

  int myints[5];

  myints[0] = 10;                //   myints: 10 20 80 320 1024
  myints[1] = 20;
  myints[2] = 80;
  myints[3] = 320;
  myints[4] = 1024;

  pair<vector<int>::iterator,int*> mypair;

  // using default comparison:
  mypair = mismatch (myvector.begin(), myvector.end(), myints);
  cout << "First mismatching elements: " << *mypair.first;
  cout << " and " << *mypair.second << endl;;

  mypair.first++; mypair.second++;

  // using predicate comparison:
  mypair = mismatch (mypair.first, myvector.end(), mypair.second, mypredicate);
  assert(*mypair.first == 40);
  assert(*mypair.second != 320);
  cout << "Second mismatching elements: " << *mypair.first;
  cout << " and " << *mypair.second << endl;;
  
  return 0;
}
