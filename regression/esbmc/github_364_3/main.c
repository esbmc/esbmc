#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#ifndef uint32_t
#define uint32_t unsigned int
#endif
typedef union
{
  uint32_t field1;
  uint32_t field2;
} str1_t;

str1_t global_str = {1};

struct str2_s
{
  bool field1;
  str1_t field2;
};

void bar(struct str2_s *str2_p)
{
  str2_p->field2 = global_str;
}

int main(void)
{
  struct str2_s str2_info;
  bar(&str2_info);
  assert(
    str2_info.field2.field1 == 2); // change to 1 for verification successful
}

