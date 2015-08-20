// map::find
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymm;

  mymm.insert (pair<char,int>('x',10));
  mymm.insert (pair<char,int>('y',20));
  mymm.insert (pair<char,int>('z',30));
  mymm.insert (pair<char,int>('z',40));

  std::multimap<char,int>::iterator it = mymm.find('x');
  assert(it->first == 'x');
  assert(it->second == 10);
  mymm.erase (it);
  mymm.erase (mymm.find('z'));

  return 0;
}
