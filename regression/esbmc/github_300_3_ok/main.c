typedef union{
    unsigned int a;
    unsigned int b;
} U1;

U1 u1 = {1};

struct S1 {
    U1 a;
};

int main(void) {
    struct S1 s1;
    struct S1 *s1p = &s1;

    s1p->a = u1;

    assert(s1.a.a == 1); // change to 2 for verification failed
}
