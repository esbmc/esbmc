// transform algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class InIt, class OutIt, class Fn1>
OutIt transform(InIt first, InIt last, OutIt dest, Fn1 func) {
	while (first != last)
		*dest++ = func(*first++);
	return dest;
}

template<class InIt1, class InIt2, class OutIt, class Fn2>
OutIt transform(InIt1 first1, InIt1 last1, InIt2 first2, OutIt dest, Fn2 func) {
	while (first1 != last1){
		*dest = func(*first1, *first2);
		dest++;
		first1++;
		first2++;
		}
	return dest;
}

int op_increase (int i) { return ++i; }
int op_sum (int i, int j) { return i+j; }

int main () {
  vector<int> first;
  vector<int> second;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<6; i++) first.push_back (i*10); //  first: 10 20 30 40 50

  second.resize(first.size());     // allocate space
  transform (first.begin(), first.end(), second.begin(), op_increase);
                                                  // second: 11 21 31 41 51

  transform (first.begin(), first.end(), second.begin(), first.begin(), op_sum);
                                                  //  first: 21 41 61 81 101

  assert(first.size() == 5);
  assert(first[0] == 21);
  assert(first[1] == 41);
  assert(first[2] != 61);
  assert(first[3] == 81);
  assert(first[4] == 101);
  cout << "first contains:";
  for (it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
