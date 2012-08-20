// copy_backward example
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

template<class BidIt1, class BidIt2>
BidIt2 copy_backward_esbmc(BidIt1 first, BidIt1 last, BidIt2 dest) {
	while (last != first)
		*(--dest) = *(--last);
	return dest;
}

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<=5; i++)
    myvector.push_back(i*10);          // myvector: 10 20 30 40 50

  myvector.resize(myvector.size()+3);  // allocate space for 3 more elements

  copy_backward_esbmc ( myvector.begin(), myvector.begin()+5, myvector.end() );
  assert(myvector[3] == 10);
  assert(myvector[7] != 50);
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
