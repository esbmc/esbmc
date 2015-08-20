// deque::rbegin/rend
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> mydeque;
  deque<int>::reverse_iterator rit;

  for (int i=1; i<=5; i++) mydeque.push_back(i);
  cout << "mydeque contains:";

  rit = mydeque.rbegin();
  assert(*rit != 5);
  int n = 5;
  while ( rit < mydeque.rend() )
  {
    cout << " " << *rit;
    assert(*rit != n);
    ++rit;
    n--;    
  }

  cout << endl;

  return 0;
}
