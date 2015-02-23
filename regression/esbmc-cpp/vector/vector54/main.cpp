// vector::pop_back
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;
  int sum (0);
  myvector.push_back (100);
  myvector.push_back (200);
  myvector.push_back (300);
  int test = 300;

  while (!myvector.empty())
  {
    sum+=myvector.back();
	 cout << test << endl;
	 assert(myvector.back() != test);
	 test += -100;
    myvector.pop_back();
  }

  cout << "The elements of myvector summed " << sum << endl;

  return 0;
}
