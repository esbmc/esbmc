#include <assert.h>
#include <stddef.h>
#include <string.h>

typedef unsigned char u1;
typedef unsigned short u2;

/* 16-byte struct */
typedef struct {
    u2 class_id;
    u2 super_class_id;
    u2 instance_size;
    u2 first_field;
    u2 num_fields;
    u2 first_method;
    u2 num_methods;
    u1 flags;
    u1 context_id;
} class_info_t;

/* 8-byte struct (different size!) */
typedef struct {
    u2 class_id;
    u2 code_offset;
    u1 max_stack;
    u1 max_locals;
    u1 num_args;
    u1 flags;
} method_info_t;

/* Parent struct with arrays of DIFFERENT struct types */
typedef struct {
    class_info_t classes[4];   /* Array of 16-byte structs */
    u2 num_classes;
    method_info_t methods[4];  /* Array of 8-byte structs */
    u2 num_methods;
} container_t;

/* This function triggers the bug */
u2 add_class(container_t* c, const class_info_t* info) {
    if (c->num_classes >= 4) return (u2)-1;
    u2 idx = c->num_classes++;
    /* BUG: This line fails with "Oversized field offset" */
    c->classes[idx] = *info;
    c->classes[idx].class_id = idx;
    return idx;
}

int main(void) {
    container_t c = {0};
    memset(&c, 0, sizeof(c));
    c.num_classes = 1;

    class_info_t cls = {
        .class_id = 0,
        .super_class_id = (u2)-1,
        .instance_size = 4,
        .first_field = 0,
        .num_fields = 1,
        .first_method = 0,
        .num_methods = 0,
        .flags = 0,
        .context_id = 0
    };

    u2 idx = add_class(&c, &cls);
    assert(idx == 1);
    assert(c.classes[1].class_id == 1);

    return 0;
}
