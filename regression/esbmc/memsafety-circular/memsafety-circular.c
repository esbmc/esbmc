
static struct item {
	struct item *next;
} *A;

int main()
{
	A = malloc(sizeof(*A));
	A->next = A;
}
