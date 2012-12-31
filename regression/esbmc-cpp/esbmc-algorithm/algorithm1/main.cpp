// for_each example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

void myfunction (int i) {
  cout << " " << i;
}

struct myclass {
  void operator() (int i) {cout << " " << i;}
} myobject;

int main () {
  vector<int> myvector;
  myvector.push_back(10);
  myvector.push_back(20);
  myvector.push_back(30);

  cout << "myvector contains:";
  for_each (myvector.begin(), myvector.end(), myfunction);

  // or:
  cout << "\nmyvector contains:";
  for_each (myvector.begin(), myvector.end(), myobject);

  cout << endl;

  return 0;
}
