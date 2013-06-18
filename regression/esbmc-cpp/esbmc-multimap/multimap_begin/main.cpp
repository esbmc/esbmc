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

  std::multimap<char,int>::iterator it = mymultimap.begin();
 
  assert((*it).first == 'a');
  assert((*it).second == 10);
  it++;
  assert((*it).first == 'b');
  assert((*it).second == 20);
  it++;
  assert((*it).first == 'b');
  assert((*it).second == 150);

  // show content:
  for (it=mymultimap.begin(); it!=mymultimap.end(); ++it)
    std::cout << (*it).first << " => " << (*it).second << '\n';

  return 0;
}

