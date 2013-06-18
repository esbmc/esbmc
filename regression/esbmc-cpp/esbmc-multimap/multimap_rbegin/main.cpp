// multimap::rbegin/rend
#include <iostream>
#include <map>
#include <cassert>

int main ()
{
  std::multimap<char,int> mymultimap;

  mymultimap.insert (std::make_pair('x',10));
  mymultimap.insert (std::make_pair('y',20));
  mymultimap.insert (std::make_pair('y',150));
  mymultimap.insert (std::make_pair('z',9));

  std::multimap<char,int>::reverse_iterator rit = mymultimap.rbegin();

  assert(rit->first == 'z');
  assert(rit->second == 9);

  for (rit=mymultimap.rbegin(); rit!=mymultimap.rend(); ++rit)
    std::cout << rit->first << " => " << rit->second << '\n';

  return 0;
}
