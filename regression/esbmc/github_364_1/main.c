// standard h files
#include <stdbool.h>
#include <stdint.h>

typedef union
{
  uint32_t raw32;

  struct
  {
    uint32_t field1 : 1;
    uint32_t field2 : 7;
    uint32_t field3 : 1;
    uint32_t field4 : 1;
    uint32_t field5 : 1;
    uint32_t field6 : 1;
    uint32_t field7 : 20;
  };
} type1_t;

struct struct2_s
{
  uint64_t field1;
  bool field2;
  bool field3;
  bool field4;
  uint16_t field5;
  type1_t field6;
  uint32_t field7;
  uint32_t field8;
  uint32_t field9;
  uint32_t field10;
  uint8_t field11;
};

void func2(type1_t param)
{
}

void func1(struct struct2_s *str2_var_p)
{
  bool bool_var = str2_var_p->field2;
  uint64_t field1 = str2_var_p->field1;

  func2(str2_var_p->field6);
}

int main(void)
{
  struct struct2_s str2_var;
  func1(&str2_var);
  return 0;
}

