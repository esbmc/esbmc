// min example
#include <iostream>
#include <cassert>
using namespace std;


const int min(const int left, const int right) {
	if (left < right)
		return left;
	else
		return right;
}

const int min(const double left, const double right) {
	if (left < right)
		return left;
	else
		return right;
}

const char min(const char left, const char right) {
	if (left < right)
		return left;
	else
		return right;
}

template<class T>
const T& min(const T& left, const T& right) {
	if (left < right)
		return left;
	else
		return right;
}

int main () {
  cout << "min(1,2)==" << min(1,2) << endl;
  assert(min(1,2) == 1);
  cout << "min(2,1)==" << min(2,1) << endl;
  cout << "min('a','z')==" << min('a','z') << endl;
  cout << "min(3.14,2.72)==" << min(3.14,2.72) << endl;
  return 0;
}
