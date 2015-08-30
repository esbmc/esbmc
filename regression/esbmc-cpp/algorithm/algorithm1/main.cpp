// for_each example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

void myfunction (int& i) {
  i++;
  cout << " " << i;
}

struct myclass {
  void operator() (int& i) {i--; cout << " " << i;}
} myobject;

int main () {
  vector<int> myvector;
  myvector.push_back(10);
  myvector.push_back(20);
  myvector.push_back(30);

  cout << "myvector contains:";
  for_each (myvector.begin(), myvector.end(), myfunction);
  assert(myvector[0] == 11);
  assert(myvector[1] == 21);
  assert(myvector[2] == 31);
  
  // or:
  cout << "\nmyvector contains:";
  for_each (myvector.begin(), myvector.end(), myobject);
  assert(myvector[0] == 10);
  assert(myvector[1] == 20);
  assert(myvector[2] == 30);

  cout << endl;

  return 0;
}
