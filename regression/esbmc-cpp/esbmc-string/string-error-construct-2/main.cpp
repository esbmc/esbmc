#include <string>
#include <cassert>
#include <iostream>
using namespace std;

int main(){
	string tmp("esbmc");
	string str(tmp, 7);
	cout << str << endl;
   assert(str.size() == 10);
	return 0;
}
