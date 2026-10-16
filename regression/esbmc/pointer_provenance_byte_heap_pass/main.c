/* Pointer values stored in heap byte storage must survive reads and realloc. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void *header_push(void *array, void *element, size_t elem_size) {
    if (array == NULL) {
        size_t *block = (size_t *)malloc(sizeof(size_t) + elem_size);
        __ESBMC_assume(block != 0);
        block[0] = 1;
        void *data = (void *)(block + 1);
        memcpy(data, element, elem_size);
        return data;
    }
    size_t old_len = ((size_t *)array)[-1];
    size_t new_len = old_len + 1;
    size_t *old_block = ((size_t *)array) - 1;
    size_t *new_block = (size_t *)realloc(
        old_block, sizeof(size_t) + new_len * elem_size);
    __ESBMC_assume(new_block != 0);
    new_block[0] = new_len;
    void *new_data = (void *)(new_block + 1);
    memcpy((char *)new_data + old_len * elem_size, element, elem_size);
    return new_data;
}

int main(void) {
    /* HEAP-allocated outer pointer-array -- the trigger.
     * Replace this with `void *flags[3] = {0};` to flip to PASS. */
    void **flags = (void **)malloc(3 * sizeof(void *));
    __ESBMC_assume(flags != 0);
    flags[0] = 0;
    flags[1] = 0;
    flags[2] = 0;

    _Bool v0 = 0, v1 = 1;
    flags[1] = header_push(flags[1], &v0, sizeof(_Bool));
    flags[1] = header_push(flags[1], &v1, sizeof(_Bool));

    _Bool *arr = (_Bool *)flags[1];
    assert(arr != NULL);
    assert(arr[0] == 0);
    assert(arr[1] == 1);
    return 0;
}
