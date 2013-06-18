// erasing from map
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymultimap;

  // insert some values:
  mymultimap.insert(std::pair<char,int>('a',10));
  mymultimap.insert(std::pair<char,int>('b',20));
  mymultimap.insert(std::pair<char,int>('b',30));
  mymultimap.insert(std::pair<char,int>('c',40));
  mymultimap.insert(std::pair<char,int>('d',50));

  std::multimap<char,int>::iterator it = mymultimap.find('b');

  mymultimap.erase (it);                     // erasing by iterator (1 element)

  mymultimap.erase ('b');                    // erasing by key (2 elements)

  it=mymultimap.find ('c');
  mymultimap.erase ( it, mymultimap.end() ); // erasing by range
  assert(mymultimap.size() == 1);
  it = mymultimap.begin();
  assert(it->first == 'a');
  assert(it->second == 10);

  return 0;
}
