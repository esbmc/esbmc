// union U;

struct S {
	union U *u;
};

union U {
	struct S s;
};

int main()
{
	union U u;
	u.s.u = &u;
}
