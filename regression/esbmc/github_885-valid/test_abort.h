
typedef union
{
  unsigned int uint_val;

  float fl_val;

} float_u;

typedef struct

{
  float_u s1;

} struct_1_t;

struct_1_t struct_1_var = {

  .s1.fl_val = 0,

};

typedef enum
{

  ENUM_1_1,

  ALL_ENUMS_1

} enum1_e;

typedef enum

{

  ENUM_2_1,

  ENUM_2_2 = ENUM_2_1,

  ALL_ENUM2_2

} enum2_e;

typedef struct
{
  enum2_e s2;

} struct_2_t;

typedef struct

{
  const struct_2_t s3[ALL_ENUMS_1];

} struct_3_t;

extern struct_3_t struct3_var;

void func2(enum1_e int_id)

{
  int x1 = struct3_var.s3[int_id].s2;
}
