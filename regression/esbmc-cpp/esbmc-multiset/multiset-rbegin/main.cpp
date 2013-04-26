#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int myints[] = {21,64,17,78,49};
  multiset<int> myset (myints,myints+5);

  multiset<int>::reverse_iterator rit;
  rit = myset.rbegin();
  assert(*rit == 78);
  
  
  cout << endl;

  return 0;
}
