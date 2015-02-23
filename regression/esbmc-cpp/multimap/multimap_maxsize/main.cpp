// multimap::max_size
#include <iostream>
#include <map>

int main ()
{
  std::multimap<int,int> mymultimap;

  if (mymultimap.max_size() >= 1000u)
  {
    for (int i=0; i<1000; i++)
      mymultimap.insert(std::make_pair(i,0));
    std::cout << "The multimap contains 1000 elements.\n";
  }
  else std::cout << "The multimap could not hold 1000 elements.\n";

  return 0;
}
