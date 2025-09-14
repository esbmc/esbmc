#include <stdlib.h>
#include <stddef.h>

struct empty_struct {
    // Completely empty
};

struct zero_width_bitfield {
    int : 0;  // Zero-width bitfield
};

struct mixed_zero_size {
    int before;
    struct empty_struct empty1;
    struct zero_width_bitfield empty2;
    struct empty_struct empty3;
    int after;
};

struct zero_array {
    int header;
    char data[0];  // Zero-length array (GCC extension)
};

int main() {
    struct mixed_zero_size *ptr = malloc(sizeof *ptr);
    ptr->before = 100;
    ptr->after = 200;
    
    // Test offsetof with zero-sized members
    size_t offset_empty1 = offsetof(struct mixed_zero_size, empty1);
    size_t offset_empty2 = offsetof(struct mixed_zero_size, empty2);
    size_t offset_empty3 = offsetof(struct mixed_zero_size, empty3);
    size_t offset_after = offsetof(struct mixed_zero_size, after);
    
    // Container_of with zero-sized member
    struct empty_struct *empty_ptr = &ptr->empty2;
    void *tmp = ((void*)empty_ptr) - offsetof(struct mixed_zero_size, empty2);
    struct mixed_zero_size *recovered = (struct mixed_zero_size*)tmp;
    
    int before = recovered->before;
    int after = recovered->after;
    
    // Test zero-length array
    struct zero_array *zarr = malloc(sizeof *zarr + 10);
    zarr->header = 0xABCD;
    
    char *data_ptr = zarr->data;
    void *tmp2 = ((void*)data_ptr) - offsetof(struct zero_array, data);
    struct zero_array *recovered2 = (struct zero_array*)tmp2;
    
    int header = recovered2->header;
    
    free(ptr);
    free(zarr);
    return 0;
}

