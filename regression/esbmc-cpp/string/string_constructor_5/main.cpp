#include <string>
#include <cassert>

using namespace std;

int main(){
	int i = 4;
	string S = string("Testing", i);
	string T = string("TestingTesting", i);
	assert(T == S);
}
