/* Solidity dynamic array operations */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include "solidity_types.h"

__attribute__((annotate("__ESBMC_inf_size"))) void *esbmc_array_ptrs[1];
__attribute__((annotate("__ESBMC_inf_size"))) size_t esbmc_array_lengths[1];
unsigned int esbmc_array_count;

void _ESBMC_array_null_check(int ok) {
__ESBMC_HIDE:;
    if (!ok)
        assert(!"Null Array Pointer");
}

void _ESBMC_element_null_check(int ok) {
__ESBMC_HIDE:;
    if (!ok)
        assert(!"Null Element Pointer");
}

void _ESBMC_zero_size_check(int ok) {
__ESBMC_HIDE:;
    if (!ok)
        assert(!"Zero Element Size");
}

void _ESBMC_pop_empty_check(int ok) {
__ESBMC_HIDE:;
    if (!ok)
        assert(!"Pop From Empty Array");
}

void _ESBMC_store_array(void *array, size_t length) {
__ESBMC_HIDE:;
    _ESBMC_array_null_check(array != 0);

    for (unsigned int i = 0; i < esbmc_array_count; ++i) {
        if (esbmc_array_ptrs[i] == array) {
            esbmc_array_lengths[i] = length;
            return;
        }
    }

    esbmc_array_ptrs[esbmc_array_count] = array;
    esbmc_array_lengths[esbmc_array_count] = length;
    esbmc_array_count++;
}

unsigned int _ESBMC_array_length(void *array) {
__ESBMC_HIDE:;
    if (array == NULL)
        return 0;

    for (unsigned int i = 0; i < esbmc_array_count; ++i) {
        if (esbmc_array_ptrs[i] == array)
            return esbmc_array_lengths[i];
    }

    // not registered
    return 0;
}

void *_ESBMC_arrcpy(void *from_array, size_t from_size, size_t size_of) {
__ESBMC_HIDE:;
    _ESBMC_element_null_check(from_array != 0);
    _ESBMC_zero_size_check(size_of != 0);
    _ESBMC_zero_size_check(from_size != 0);
    size_t bytes = from_size * size_of;
    void *to_array = malloc(bytes);
    __builtin_memcpy(to_array, from_array, bytes);

    _ESBMC_store_array(to_array, from_size);
    return to_array;
}

void *_ESBMC_array_push(void *array, void *element, size_t size_of_element) {
__ESBMC_HIDE:;
    _ESBMC_zero_size_check(size_of_element != 0);

    char *fallback_zero = NULL;
    if (element == NULL) {
        fallback_zero = (char *)calloc(1, size_of_element);
        element = fallback_zero;
    }

    // Case 1: array is NULL (new array allocation)
    if (array == NULL) {
        void *new_array = malloc(size_of_element);
        for (size_t j = 0; j < size_of_element; ++j)
            ((char *)new_array)[j] = ((char *)element)[j];

        _ESBMC_store_array(new_array, 1);

        if (fallback_zero != NULL)
            free(fallback_zero);
        return new_array;
    }

    // Case 2: array already registered
    for (unsigned int i = 0; i < esbmc_array_count; ++i) {
        if (esbmc_array_ptrs[i] == array) {
            size_t old_len = esbmc_array_lengths[i];
            size_t new_len = old_len + 1;
            void *new_array = realloc(array, new_len * size_of_element);

            for (size_t j = 0; j < size_of_element; ++j)
                ((char *)new_array)[old_len * size_of_element + j] =
                    ((char *)element)[j];

            esbmc_array_ptrs[i] = new_array;
            esbmc_array_lengths[i] = new_len;

            if (fallback_zero != NULL)
                free(fallback_zero);
            return new_array;
        }
    }

    // Case 3: array is non-NULL but not tracked (edge case fallback)
    void *new_array = malloc(size_of_element);
    for (size_t j = 0; j < size_of_element; ++j)
        ((char *)new_array)[j] = ((char *)element)[j];
    _ESBMC_store_array(new_array, 1);

    if (fallback_zero != NULL)
        free(fallback_zero);
    return new_array;
}


void _ESBMC_array_pop(void *array, size_t size_of_element) {
__ESBMC_HIDE:;
    _ESBMC_array_null_check(array != 0);
    _ESBMC_zero_size_check(size_of_element != 0);

    for (unsigned int i = 0; i < esbmc_array_count; ++i) {
        if (esbmc_array_ptrs[i] == array) {
            _ESBMC_pop_empty_check(esbmc_array_lengths[i] > 0);

            esbmc_array_lengths[i]--;

            if (esbmc_array_lengths[i] == 0) {
                free(esbmc_array_ptrs[i]);
                esbmc_array_ptrs[i] = 0;
            } else {
                void *new_array = realloc(esbmc_array_ptrs[i], esbmc_array_lengths[i] * size_of_element);
                if (new_array != 0)
                    esbmc_array_ptrs[i] = new_array;
            }

            return;
        }
    }

    _ESBMC_pop_empty_check(0); // uninitialized array pop
}
