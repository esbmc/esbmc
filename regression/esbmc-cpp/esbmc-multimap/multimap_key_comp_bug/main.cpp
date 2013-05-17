// map::key_comp
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymultimap;

  std::multimap<char,int>::key_compare mycomp = mymultimap.key_comp();

  mymultimap.insert (std::make_pair('a',100));
  mymultimap.insert (std::make_pair('b',200));
  mymultimap.insert (std::make_pair('b',211));
  mymultimap.insert (std::make_pair('c',300));

  std::cout << "mymultimap contains:\n";

  char highest = mymultimap.rbegin()->first;     // key value of last element

  std::multimap<char,int>::iterator it = mymultimap.begin();
  assert(!mycomp((*it).first, highest));
  do {
    std::cout << (*it).first << " => " << (*it).second << '\n';
  } while ( mycomp((*it++).first, highest) );

  std::cout << '\n';

  return 0;
}
