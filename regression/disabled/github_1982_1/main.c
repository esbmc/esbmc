#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct vector vector;

#define VECTOR_INIT_CAPACITY 10
#define VECTOR_FACTOR 2

struct vector {
  size_t length;
  size_t _capacity;
  size_t _size_of_element;
  void *_buf;
};

vector vector_init(const size_t unit_length) {
  vector v;
  v._size_of_element = unit_length;
  v.length = 0;
  v._capacity = VECTOR_INIT_CAPACITY;
  v._buf = malloc(unit_length * VECTOR_INIT_CAPACITY);

  return v;
}

void vector_destroy(vector * const v) {
  if (v->_buf != NULL)
    free(v->_buf);
}

void vector_reserve(vector * const v, const size_t quantity) {
  assert(v->_capacity < quantity); // We should be increasing
#ifdef __ESBMC
  char *buf = malloc(quantity * v->_size_of_element);
  memcpy(buf, v->_buf, v->_capacity);
  v->_capacity = quantity;
  free(v->_buf);
  v->_buf = buf;
#else
  v->_capacity = quantity;
  v->_buf = realloc(v->_buf, v->_capacity * v->_size_of_element);
  #endif
}

void vector_push_back(vector * const v, const void * const elem) {
  if (v->length == v->_capacity)
    vector_reserve(v, v->_capacity * VECTOR_FACTOR);

  void *addr =  v->_buf + v->length++ * v->_size_of_element;
  memcpy(addr, elem, v->_size_of_element);
}

void *vector_at(vector *const v, const size_t i) {
  assert(i < v->length);
  return (void*)(v->_buf) + (i * v->_size_of_element);
}

void vector_test() {
  size_t length = 25;
  char arr[length];
  for (char i = 0; i < length; i++)
    arr[i] = i;

  printf("\t- Testing vectors\n");
  vector v = vector_init(sizeof(char));

  for (int i = 0; i < length; i++)
    vector_push_back(&v, &arr[i]);
  assert(v.length == length);
  for (int i = 0; i < length; i++)
    assert(*(char*)vector_at(&v, i) == i);
  vector_destroy(&v);
}


int main() {
  vector_test();
}
