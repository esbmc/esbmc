#include <stdlib.h>
#include <stdint.h>

// Test function to verify pointer object simplification
void test_pointer_object_simplification() 
{
    int *base_ptr = malloc(sizeof(int) * 10);
    char *char_ptr = (char*)base_ptr;
    
    // pointer_object(ptr + offset) should equal pointer_object(ptr)
    uintptr_t obj1 = __ESBMC_POINTER_OBJECT(base_ptr);
    uintptr_t obj2 = __ESBMC_POINTER_OBJECT(base_ptr + 5);
    
    __ESBMC_assert(obj1 == obj2, 
        "Pointer arithmetic: pointer_object(ptr + 5) should equal pointer_object(ptr)");
    
    // pointer_object((int*)ptr) should equal pointer_object(ptr)
    uintptr_t obj3 = __ESBMC_POINTER_OBJECT(char_ptr);
    uintptr_t obj4 = __ESBMC_POINTER_OBJECT((int*)char_ptr);
    
    __ESBMC_assert(obj3 == obj4,
        "Typecast: pointer_object((int*)ptr) should equal pointer_object(ptr)");
    
    // pointer_object(((char*)ptr) + offset) should equal pointer_object(ptr)
    uintptr_t obj5 = __ESBMC_POINTER_OBJECT(base_ptr);
    uintptr_t obj6 = __ESBMC_POINTER_OBJECT(((char*)base_ptr) + 20);
    
    __ESBMC_assert(obj5 == obj6,
        "Combined: pointer_object(((char*)ptr) + 20) should equal pointer_object(ptr)");
    
    // pointer_object((ptr + 3) + 7) should equal pointer_object(ptr)
    int *temp_ptr = base_ptr + 3;
    uintptr_t obj7 = __ESBMC_POINTER_OBJECT(base_ptr);
    uintptr_t obj8 = __ESBMC_POINTER_OBJECT(temp_ptr + 7);
    
    __ESBMC_assert(obj7 == obj8,
        "Nested arithmetic: pointer_object((ptr + 3) + 7) should equal pointer_object(ptr)");
    
    // pointer_object((int*)(((char*)ptr) + 8)) should equal pointer_object(ptr)
    uintptr_t obj9 = __ESBMC_POINTER_OBJECT(base_ptr);
    uintptr_t obj10 = __ESBMC_POINTER_OBJECT((int*)(((char*)base_ptr) + 8));
    
    __ESBMC_assert(obj9 == obj10,
        "Complex: pointer_object((int*)(((char*)ptr) + 8)) should equal pointer_object(ptr)");
        
    // pointer_object(ptr - 2) should equal pointer_object(ptr)  
    uintptr_t obj11 = __ESBMC_POINTER_OBJECT(base_ptr);
    uintptr_t obj12 = __ESBMC_POINTER_OBJECT(base_ptr - 2);
    
    __ESBMC_assert(obj11 == obj12,
        "Negative offset: pointer_object(ptr - 2) should equal pointer_object(ptr)");
    
    free(base_ptr);
}

// Additional test with different allocation types
void test_different_allocations()
{
    // Stack allocation
    int stack_array[10];
    uintptr_t stack_obj1 = __ESBMC_POINTER_OBJECT(stack_array);
    uintptr_t stack_obj2 = __ESBMC_POINTER_OBJECT(stack_array + 3);
    
    __ESBMC_assert(stack_obj1 == stack_obj2,
        "Stack array: pointer_object(array + 3) should equal pointer_object(array)");
    
    // Global/static allocation  
    static char static_buffer[100];
    uintptr_t static_obj1 = __ESBMC_POINTER_OBJECT(static_buffer);
    uintptr_t static_obj2 = __ESBMC_POINTER_OBJECT((int*)(static_buffer + 16));
    
    __ESBMC_assert(static_obj1 == static_obj2,
        "Static buffer: pointer_object((int*)(buffer + 16)) should equal pointer_object(buffer)");
}

// Test to verify different objects have different pointer objects
void test_different_objects()
{
    int *ptr1 = malloc(sizeof(int));
    int *ptr2 = malloc(sizeof(int)); 
    
    uintptr_t obj1 = __ESBMC_POINTER_OBJECT(ptr1);
    uintptr_t obj2 = __ESBMC_POINTER_OBJECT(ptr2);
    
    // Different allocations should have different pointer objects
    __ESBMC_assert(obj1 != obj2,
        "Different allocations should have different pointer objects");
    
    // But arithmetic on same object should still match
    uintptr_t obj1_offset = __ESBMC_POINTER_OBJECT(ptr1 + 0); // Adding 0 to test edge case
    __ESBMC_assert(obj1 == obj1_offset,
        "Same object with zero offset should have same pointer object");
    
    free(ptr1);
    free(ptr2);
}

int main()
{
    test_pointer_object_simplification();
    test_different_allocations(); 
    test_different_objects();
    
    return 0;
}
