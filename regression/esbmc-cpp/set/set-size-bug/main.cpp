#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int b[9] = {1,2,3,4,5,6,7,8,9};
  set<int> myints(b,b+9);
  assert(myints.size() != 9);
  
  

  return 0;
}
