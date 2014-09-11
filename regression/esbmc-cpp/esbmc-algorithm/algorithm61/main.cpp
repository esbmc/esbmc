// min_element/max_element
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;

bool myfn(int i, int j) { return i<j; }

struct myclass {
  bool operator() (int i,int j) { return i<j; }
} myobj;

int main () {
  int myints[] = {3,7,2,5,6,4,9};

  // using default comparison:
  cout << "The smallest element is " << *min_element(myints,myints+7) << endl;
  assert(*min_element(myints,myints+7) == 2);
  cout << "The largest element is " << *max_element(myints,myints+7) << endl;
  assert(*max_element(myints,myints+7) == 9);

  // using function myfn as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myfn) << endl;
  cout << "The largest element is " << *max_element(myints,myints+7,myfn) << endl;

  // using object myobj as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myobj) << endl;
  cout << "The largest element is " << *max_element(myints,myints+7,myobj) << endl;

  return 0;
}
