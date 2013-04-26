#include <iostream>
#include <set>
#include <cassert>
using namespace std;

int main ()
{
  int b[9] = {10,20,30,40,50,60,70,80,90};
  multiset<int> myset(b,b+9);
  multiset<int>::iterator it;
  int i;

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
  
  it = myset.begin();
  assert(*it==10);
  it++;
  assert(*it==30);
  it++;
  assert(*it!=50);

  cout << endl;

  return 0;
}
