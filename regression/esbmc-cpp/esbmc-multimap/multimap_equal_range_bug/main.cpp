// map::equal_elements
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymm;

  mymm.insert(std::pair<char,int>('a',10));
  mymm.insert(std::pair<char,int>('b',20));
  mymm.insert(std::pair<char,int>('b',40));
  mymm.insert(std::pair<char,int>('c',50));
  mymm.insert(std::pair<char,int>('d',60));

  std::pair <std::multimap<char,int>::iterator, std::multimap<char,int>::iterator> ret;
  ret = mymm.equal_range('b');

  assert(ret.first->second == 20);
  assert(ret.second->second == 40);

  return 0;
}
