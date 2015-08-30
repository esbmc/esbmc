#include <iostream>
#include <cassert>
#include <set>
using namespace std;

int main ()
{
  int myints[] = {78,21,64,49,17};
  set<int> myset (myints,myints+5);

  set<int>::reverse_iterator rit;
  rit = myset.rend();
  rit++;
  assert(*rit == 78);
  cout << "myset contains:";
//  for ( rit=myset.rbegin() ; rit != myset.rend(); rit++ )
    cout << " " << *rit;

  cout << endl;

  return 0;
}
