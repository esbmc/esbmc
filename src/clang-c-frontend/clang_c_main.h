#ifndef CLANG_C_FRONTEND_CLANG_C_MAIN_H_
#define CLANG_C_FRONTEND_CLANG_C_MAIN_H_

#include <util/context.h>
#include <util/message.h>
#include <util/std_code.h>

class clang_c_maint
{
public:
  clang_c_maint(contextt &_context) : context(_context)
  {
  }

  bool clang_c_main();
  void init_variable(codet &dest, const symbolt &sym);
  void static_lifetime_init(const contextt &context, codet &dest);

protected:
  contextt &context;
};

#endif /* CLANG_C_FRONTEND_CLANG_C_MAIN_H_ */
