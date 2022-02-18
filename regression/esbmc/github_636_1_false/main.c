// Testing allocation with a nondet number

int bar(void *ptr) {
  unsigned int size = __ESBMC_get_object_size(ptr); // Do we have something like this?
  __ESBMC_assert(size < 40, "Allocated size should be equal");
}

int main() {
    unsigned int alloc_size = nondet_uint();
    __ESBMC_assume(alloc_size > 0 && alloc_size < 60);
    void * foo = malloc(alloc_size);
    bar(foo); 
}