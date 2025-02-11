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
	B.c = &B;
	assert(B.c == B.c->c);
}
