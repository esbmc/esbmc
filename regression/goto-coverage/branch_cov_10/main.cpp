#include <stack>
#include <string>
#include <cassert>
using namespace std;

template<class T>
	class A
{
public:
	A(){
		this->a = true;
	}
	bool get(){
		return this->a;
	}
	void set( bool x ){
		this->a = x;
	}
private:
	bool a;
};

bool operator!=(A<bool> x, A<bool> y){
	if(x.get() != y.get())
		return true;
	return false;
}

bool operator==(A<bool> x, A<bool> y){
	if(x.get() == y.get())
		return true;
	return false;
}

bool operator<(A<bool> x, A<bool> y){
	return false;
}

int main ()
{
  stack< A<bool> > first;
  A< bool > x;
  A< bool > y;
  A< bool > z;
  first.push(x);
  first.push(y);
  first.push(z);
  first.pop();
  assert(!(first.empty()));
  assert((first.top()).get());
  assert(first.size() == 2);
  return 0;
}
