typedef union {
	struct { int z; } x;
} t1;

typedef struct {
	int y[2];
} t2;

t1 v1;
t2 v2;

int main()
{
	v1 = *(t1 *)&v2;
}
