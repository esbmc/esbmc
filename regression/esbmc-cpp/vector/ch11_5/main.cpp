// vector::size
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;

  for (int i=0; i<10; i++) myints.push_back(i);
  cout << "1. size: " << (int) myints.size() << endl;

  return 0;
}
