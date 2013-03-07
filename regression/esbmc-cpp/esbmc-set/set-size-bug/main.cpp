#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int b[10] = {0,1,2,3,4,5,6,7,8,9};
  set<int> myints(b,b+10);
  assert(myints.size() != 10);
  
  

  return 0;
}
