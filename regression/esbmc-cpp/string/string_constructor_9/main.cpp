#include <string>
#include <cassert>

using namespace std;

int main(){
	
	string S = string("Testing", 2);
	string S1 = string("Testing", 2);
	string T = string("Testing", 5);
	assert(T != S + S1);
}
