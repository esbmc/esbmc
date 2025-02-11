#include <cassert>

class a;
class b;
class a {
public:
  b *c;
};
class b : public a {};

int main()
{
	b B;
	a *A = &B;
	A->c = &B;
	assert(A->c == static_cast<a *>(A->c)->c);
}
