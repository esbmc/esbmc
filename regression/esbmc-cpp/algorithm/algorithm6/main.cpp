// adjacent_find example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};
  vector<int> myvector (myints,myints+8);
  vector<int>::iterator it;

  // using default comparison:
  it = adjacent_find (myvector.begin(), myvector.end());

  if (it!=myvector.end())
    cout << "the first consecutive repeated elements are: " << *it << endl;

  //using predicate comparison:
  it = adjacent_find (++it, myvector.end(), myfunction);

  if (it!=myvector.end())
    cout << "the second consecutive repeated elements are: " << *it << endl;
  
  return 0;
}
