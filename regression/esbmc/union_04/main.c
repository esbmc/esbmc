#include <stdbool.h>
#include <stdint.h>
#ifndef uint32_t
#define uint32_t unsigned int
#endif
typedef union
{
  uint32_t field1;
  struct
  {
    uint32_t field2;
  };
} str1_t;
extern str1_t global_str;
struct str2_s
{
  bool field1;
  str1_t field2;
};
void bar(struct str2_s *str2_p)
{
  str2_p->field2 = global_str;
}
void foo(void)
{
  struct str2_s str2_info;
  bar(&str2_info);
}
