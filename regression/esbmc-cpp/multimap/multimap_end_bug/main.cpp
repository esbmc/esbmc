// map::begin/end
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymultimap;

  mymultimap.insert (std::pair<char,int>('a',10));
  mymultimap.insert (std::pair<char,int>('b',20));
  mymultimap.insert (std::pair<char,int>('b',150));

  std::multimap<char,int>::iterator it = mymultimap.end();
 
  it--;
  assert((*it).first == 'b');
  assert((*it).second != 150);

  return 0;
}

