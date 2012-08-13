// vector assignment
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> first (3,0);
  vector<int> second (5,0);

  second=first;
  first=vector<int>();

  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  return 0;
}
