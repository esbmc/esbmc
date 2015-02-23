//TEST FAILS
// constructing deques
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;

  // constructors used in the same order as described above:
  deque<int> first;                                // empty deque of ints
  deque<int> second (4,100);                       // four ints with value 100
  deque<int> third (second.begin(),second.end());  // iterating through second
  deque<int> fourth (third);                       // a copy of third
  
  assert(fourth == third);
  assert(first.size() == 0);
  assert(second.size() == 4);
  assert(second[2] == 100);
  assert(third[0] == 100);

  // the iterator constructor can also be used to construct from arrays:
  int myints[] = {16,2,77,29};
  deque<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );

  cout << "The contents of fifth are:";
  for (i=0; i < fifth.size(); i++)
    assert(fifth[i] != myints[i]);

  cout << endl;

  return 0;
}
