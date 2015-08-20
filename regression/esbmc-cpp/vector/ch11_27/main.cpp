// swap vectors
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  unsigned int i;
  vector<int> first (3,100);   // three ints with a value of 100
  vector<int> second (5,200);  // five ints with a value of 200

  first.swap(second);

  cout << "first contains:";
  for (i=0; i<first.size(); i++) cout << " " << first[i];

  cout << "\nsecond contains:";
  for (i=0; i<second.size(); i++) cout << " " << second[i];

  cout << endl;

  return 0;
}
