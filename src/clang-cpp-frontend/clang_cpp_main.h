#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_

#include <clang-c-frontend/clang_c_main.h>

class clang_cpp_maint : public clang_c_maint
{
public:
  clang_cpp_maint(contextt &_context) : clang_c_maint(_context)
  {
  }

  bool clang_cpp_main();
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_ */
