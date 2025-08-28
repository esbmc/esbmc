#include <stdlib.h>
#include <stdint.h> // SIZE_MAX


//void *__ESBMC_list_append_dummy(
//  void *array,
//  size_t *len,
//  size_t elem_size,
//  void *elem)
//{
//  void *tmp = realloc(array, (*len + 1) * elem_size);
//  if (!tmp)
//    return NULL;
//  memcpy((char *)tmp + (*len * elem_size), elem, elem_size);
//  (*len)++;
//  return tmp;
//}

// Append N elements, keeping the buffer ALWAYS with the exact size.
// - array: current pointer (can be NULL when len == 0)
// - len:   current number of elements
// - elem_size: size of each element in bytes
// - src:   pointer to N contiguous elements to append
// - n:     number of elements to append
// Returns a new pointer on success (old block is freed);
// NULL on failure (old array is left untouched).
void *__ESBMC_list_append_dummy(
  void *array, size_t len, size_t elem_size, const void *src, size_t n)
{
  if (n == 0)
    return array;

  if (!src || elem_size == 0)
    return NULL;

  // Check overflow in len + n
  if (len > SIZE_MAX - n)
    return NULL;

  size_t new_len = len + n;

  // Check overflow in new_len * elem_size
  if (elem_size != 0 && new_len > SIZE_MAX / elem_size)
    return NULL;

  size_t old_bytes = len * elem_size;
  size_t new_bytes = new_len * elem_size;

  // Allocate new block with the required size
  void *newbuf = malloc(new_bytes);
  if (!newbuf)
    return NULL;

  // Copy old elements if any
  if (array && old_bytes)
  {
    memcpy(newbuf, array, old_bytes);
  }

  // Copy the new elements at the end
  memcpy((char *)newbuf + old_bytes, src, n * elem_size);

  // Free the old block
  free(array);

  return newbuf;
}
