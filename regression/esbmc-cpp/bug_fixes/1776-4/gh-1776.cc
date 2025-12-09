#include <cassert>

class a;
class b;
class a {
public:
  a *c;
};
class b : public a {};

int main()
{
	b B;
	a *A = &B;
	A->c = A;
	assert(A->c == A->c->c);
}
