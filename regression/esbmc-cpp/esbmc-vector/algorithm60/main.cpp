// max example
#include <iostream>
#include <cassert>
using namespace std;

template<class T>
const T& max(const T& left, const T& right) {
	if (left > right)
		return left;
	else
		return right;
}

const double max(const double left, const double right) {
	if (left > right)
		return left;
	else
		return right;
}

const int max(const int left, const int right) {
	if (left > right)
		return left;
	else
		return right;
}

const char max(const char left, const char right) {
	if (left > right)
		return left;
	else
		return right;
}

int main () {
  cout << "max(1,2)==" << max(1,2) << endl;
  assert(max(1,2) == 2);
  cout << "max(2,1)==" << max(2,1) << endl;
  cout << "max('a','z')==" << max('a','z') << endl;
  cout << "max(3.14,2.73)==" << max(3.14,2.73) << endl;
  return 0;
}
