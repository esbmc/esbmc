// find_end example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

int main () {
  int myints[] = {1,2,3,4,5,1,2,3,4,5};
  vector<int> myvector (myints,myints+10);
  vector<int>::iterator it;

  int match1[] = {1,2,3};

  // using default comparison:
  it = find_end (myvector.begin(), myvector.end(), match1, match1+3);
  assert(*it == 1);

  int match2[] = {4,5,1};

  // using predicate comparison:
  it = find_end (myvector.begin(), myvector.end(), match2, match2+3, myfunction);
  assert(*it == 4);
  return 0;
}
