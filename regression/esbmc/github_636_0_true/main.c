// Testing allocation with a fixed number

unsigned int MAGIC_NUMBER = 32;

int bar(void *ptr) {
  unsigned int size = __ESBMC_get_object_size(ptr); // Do we have something like this?
  __ESBMC_assert(size == (MAGIC_NUMBER), "Allocated size should be equal");
}

int main() {
    void * foo = malloc(MAGIC_NUMBER);
    bar(foo); 
}