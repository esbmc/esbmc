// list::max_size
#include <iostream>
#include <list>
using namespace std;

int main ()
{
  unsigned int i;
  list<int> mylist;

  cout << "Enter number of elements: ";
  cin >> i;

  if (i<mylist.max_size()) mylist.resize(i);
  else cout << "That size exceeds the limit.\n";

  return 0;
}
