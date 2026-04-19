#include <list>
#include <algorithm>
#include <assert.h>
#include <iostream>

int main(void)
{
  std::list<int> l;
  l.push_back(1);
  auto itr = std::find(l.begin(), l.end(), 1);
  assert(*itr == 1);
}