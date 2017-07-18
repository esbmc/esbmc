
#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_

#include <clang-c-frontend/clang_c_adjust.h>

class clang_cpp_adjust : public clang_c_adjust
{
  public:
    clang_cpp_adjust(contextt &_context);
    virtual ~clang_cpp_adjust() = default;
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_ */
