#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[9] = {10,20,30,40,50,60,70,80,90};
  set<int> myset(b,b+9);
  set<int>::iterator it;
  int i;

  // insert some values:

  it=myset.begin();
  it++;                                         // "it" points now to 20
  assert(*it == 20);
  myset.erase (it);
  it=myset.begin();
  it++;
  assert(*it == 30);
  myset.erase (40);

  it=myset.find (60);
  myset.erase ( it, myset.end() );

  cout << "myset contains:";
  for (it=myset.begin(), i = 10; it!=myset.end(); ++it, i+=20){
    cout << " " << *it;
    assert(*it != i);
  }
  cout << endl;

  return 0;
}
