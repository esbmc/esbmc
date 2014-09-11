// includes algorithm example
#include <iostream>
#include <cassert>
#include <algorithm>
using namespace std;

bool myfunction (int i, int j) { return i<j; }

int main () {
  int container[] = {5,10,15,20,25,30,35,40,45,50};
  int continent[] = {40,30,20,10};

  sort (container,container+10);
  sort (continent,continent+4);

  // using default comparison:
  if ( includes(container,container+10,continent,continent+4) )
    cout << "container includes continent!" << endl;

  assert(!(includes(container,container+10,continent,continent+4)));

  // using myfunction as comp:
  if ( includes(container,container+10,continent,continent+4, myfunction) )
    cout << "container includes continent!" << endl;
    
  assert(!(includes(container,container+10,continent,continent+4, myfunction)));

  return 0;
}
