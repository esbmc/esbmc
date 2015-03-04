// multimap::insert (C++98)
#include <iostream>
#include <map>
#include <cassert>
int main ()
{
  std::multimap<char,int> mymultimap;
  std::multimap<char,int>::iterator it;

  // first insert function version (single parameter):
  mymultimap.insert ( std::pair<char,int>('a',100) );
  mymultimap.insert ( std::pair<char,int>('z',150) );
  it=mymultimap.insert ( std::pair<char,int>('b',75) );

  assert(it->first == 'b');
  assert(it->second == 75);
  it--;
  assert(it->first == 'a');
  assert(it->second == 100);

  return 0;
}
