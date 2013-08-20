// sort algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }

struct myclass {
  bool operator() (int i,int j) { return (i<j);}
} myobject;

int main () {
  int myints[] = {32,71,12};
  int myints1[] = {12,32,71}; 
  
  vector<int> myvector (myints, myints+3);               // 32 71 12 45 26 80 53 33
  vector<int>::iterator it;

  // using default comparison (operator <):
//  sort (myvector.begin(), myvector.begin()+3);           //(12 32 45 71)26 80 53 33
  
  // using function as comp
//  sort (myvector.begin()+4, myvector.end(), myfunction); // 12 32 45 71(26 33 53 80)

  // using object as comp
  sort (myvector.begin(), myvector.end(), myobject);     //(12 26 32 33 45 53 71 80)
  assert(myvector[0] == 12);
  assert(myvector[1] == 32);
  assert(myvector[2] == 71);

  // print out content:
  
  cout << endl;

  return 0;
}
