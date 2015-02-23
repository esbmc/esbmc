// min_element/max_element
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

bool myfn(int i, int j) { return i<j; }

struct myclass {
  bool operator() (int i,int j) { return i<j; }
} myobj;

int main () {
  int myints[] = {3,7,2,5,6,4,9};

  // using default comparison:
  cout << "The smallest element is " << *min_element(myints,myints+7) << endl;
  cout << "The largest element is " << *max_element(myints,myints+7) << endl;

  // using function myfn as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myfn) << endl;
  cout << "The largest element is " << *max_element(myints,myints+7,myfn) << endl;

  // using object myobj as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myobj) << endl;
  assert(*max_element(myints,myints+7,myobj) != 9);
  cout << "The largest element is " << *max_element(myints,myints+7,myobj) << endl;

  return 0;
}
