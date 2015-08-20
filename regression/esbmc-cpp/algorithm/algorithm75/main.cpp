// search_n example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool mypredicate (int i, int j) {
  return (i==j);
}

int main () {
  int myints[]={10,20,30,30,20};
  vector<int> myvector (myints,myints+5);

  vector<int>::iterator it;

  // using default comparison:
  it = search_n (myvector.begin(), myvector.end(), 2, 30);
  
  assert(*it != 30);


  return 0;
}
