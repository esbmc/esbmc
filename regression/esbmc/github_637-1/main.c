struct S {
	struct T *t;
};

struct T {
	struct S s;
};

int main()
{
	struct T t;
	t.s.t = &t;
}
