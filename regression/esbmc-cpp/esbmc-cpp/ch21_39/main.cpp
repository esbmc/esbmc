// list::push_front
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> mylist (2,100);         // two ints with a value of 100
  mylist.push_front (200);
  mylist.push_front (300);

  std::cout << "(int) mylist.size(): " << (int) mylist.size() << std::endl;

  assert((int) mylist.size() == 4);

  return 0;
}

