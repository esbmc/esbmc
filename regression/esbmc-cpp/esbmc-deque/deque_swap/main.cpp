// swap deques
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  unsigned int i;
  deque<int> first (3,100);   // three ints with a value of 100
  deque<int> second (5,200);  // five ints with a value of 200
  
  assert(first[2] == 100);
  assert(second[4] == 200);
  
  first.swap(second);

  assert(first[4] == 200);
  assert(second[2] == 100);

  cout << "first contains:";
  for (i=0; i<first.size(); i++) cout << " " << first[i];

  cout << "\nsecond contains:";
  for (i=0; i<second.size(); i++) cout << " " << second[i];

  cout << endl;

  return 0;
}
