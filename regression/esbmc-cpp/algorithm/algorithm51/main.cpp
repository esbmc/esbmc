// includes algorithm example
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;

bool myfunction (int i, int j) { return i<j; }

int main () {
  int container[] = {5,10,15,20,25,30,35,40,45,50};
  int continent[] = {10,20,30,40};

  // using default comparison:
//  if ( includes(container,container+10,continent,continent+4) )
//    cout << "container includes continent!" << endl;
  assert(includes(container,container+10,continent,continent+4));
  assert(includes(container,container+10,continent,continent+4, myfunction));
  // using myfunction as comp:
//  if ( includes(container,container+10,continent,continent+4, myfunction) )
//    cout << "container includes continent!" << endl;

  return 0;
}
