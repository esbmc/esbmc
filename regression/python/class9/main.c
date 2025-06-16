#include <string.h>
#include <assert.h>

struct BankAccount
{
  const char *owner_;
};

#if 0
struct BankAccount init(const char *owner)
{
  struct BankAccount ba;
  ba.owner_ = owner;
  return ba;
}

int main(void)
{
  struct BankAccount b = init("Alice");
  assert(strcmp(b.owner_, "Alice") == 0);
  return 0;
}
#endif

void foo(const char *ptr)
{
  const char *copy = ptr;
//  assert(strcmp("bruno", copy) == 0);
}

int main(void)
{
  struct BankAccount ba = {.owner_ = "bruno"};

//code
//  * type: code
//  * operands:
//    0: symbol
//        * type: symbol
//            * identifier: tag-struct BankAccount
//        * name: ba
//        * identifier: c:main.c@438@F@main@ba
//    1: struct
//        * type: struct
//            * tag: struct BankAccount
//            * components:
//              0: component
//                  * type: pointer
//                      * subtype: signedbv
//                          * width: 8
//                          * #constant: 1
//                          * #cpp_type: signed_char
//                  * name: owner_
//                  * pretty_name: owner_
//                  * #location:
//                    * file: main.c
//                    * line: 6
//                    * column: 3
//            * #location:
//              * file: main.c
//              * line: 4
//              * column: 1
//        * operands:
//          0: address_of
//              * type: pointer
//                  * subtype: signedbv
//                      * width: 8
//                      * #cpp_type: signed_char
//              * operands:
//                0: index
//                    * type: signedbv
//                        * width: 8
//                        * #cpp_type: signed_char
//                    * operands:
//                      0: string-constant
//                          * type: array
//                              * size: constant
//                                  * type: unsignedbv
//                                      * width: 64
//                                  * value: 0000000000000000000000000000000000000000000000000000000000000110
//                                  * #cformat: 6
//                              * subtype: signedbv
//                                  * width: 8
//                                  * #cpp_type: signed_char
//                          * value: bruno
//                          * kind: default
//                          * #location:
//                            * file: main.c
//                            * line: 33
//                            * function: main
//                            * column: 38
//                      1: constant
//                          * type: signedbv
//                              * width: 64
//                          * value: 0000000000000000000000000000000000000000000000000000000000000000
//              * #location:
//                * file: main.c
//                * line: 33
//                * function: main
//                * column: 38
//        * #location:
//          * file: main.c
//          * line: 33
//          * function: main
//          * column: 27
//  * statement: decl
//  * #location:
//    * file: main.c
//    * line: 33
//    * function: main
//    * column: 3

  foo(ba.owner_);

//  member
//    * type: pointer
//        * subtype: signedbv
//            * width: 8
//            * #constant: 1
//            * #cpp_type: signed_char
//    * operands:
//      0: symbol
//          * type: symbol
//              * identifier: tag-struct BankAccount
//          * name: ba
//          * identifier: c:main.c@438@F@main@ba
//          * #location:
//            * file: main.c
//            * line: 34
//            * function: main
//            * column: 7
//          * #lvalue: 1
//    * component_name: owner_
//    * #location:
//      * file: main.c
//      * line: 34
//      * function: main
//      * column: 7

//  assert(strcmp("bruno", ba.owner_) == 0);
  return 0;
}
