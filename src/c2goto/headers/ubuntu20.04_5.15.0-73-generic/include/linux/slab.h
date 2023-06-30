#include "gfp.h"


void check_gfp_flags(gfp_t flags);

void *__kmalloc(size_t size, gfp_t flags);

void *__kmalloc_large(size_t size, gfp_t flags);

void *kmalloc(int size, int flags);

void kfree(const void *ptr);


