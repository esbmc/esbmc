typedef struct {
	struct T *t;
} S;

typedef struct T {
	S s;
} T;

int main()
{
	T t;
	t.s.t = &t;
}
