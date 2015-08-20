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
  // second insert function version (with hint position):
  mymultimap.insert (it, std::pair<char,int>('c',300));  // max efficiency inserting
  mymultimap.insert (it, std::pair<char,int>('z',400));  // no max efficiency inserting

  assert(it->first == 'b');
  assert(it->second == 75);
  it++;
  assert(it->first == 'c');
  assert(it->second != 300);

  return 0;
}
