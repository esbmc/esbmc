int some_var;

typedef struct local_s
{
    void *ptr;
} local_t;

void init(void* base, unsigned int size) {
    for (int i = 0; i < size; i++) {
        *((char*)base + i) = __VERIFIER_nondet_uchar();
    }
}

local_t local_data;

int main()
{
    init(&local_data, sizeof(local_t));

    assert(local_data.ptr != &some_var);
}
