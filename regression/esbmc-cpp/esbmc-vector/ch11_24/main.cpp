// vector::pop_back
#include <iostream>
#include <vector>
using namespace std;

int main ()
{
  vector<int> myvector;
  int sum (0);
  myvector.push_back (100);
  myvector.push_back (200);
  myvector.push_back (300);

  while (!myvector.empty())
  {
    sum+=myvector.back();
    myvector.pop_back();
  }

  cout << "The elements of myvector summed " << sum << endl;

  return 0;
}
