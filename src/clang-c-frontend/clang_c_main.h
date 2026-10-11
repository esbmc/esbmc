#ifndef CLANG_C_FRONTEND_CLANG_C_MAIN_H_
#define CLANG_C_FRONTEND_CLANG_C_MAIN_H_

#include <util/symtab/context.h>
#include <util/message/message.h>
#include <util/irep/std_code.h>

class clang_c_maint
{
public:
  clang_c_maint(contextt &_context) : context(_context)
  {
  }

  bool clang_main();
  void init_variable(codet &dest, const symbolt &sym);
  virtual void adjust_init(code_assignt &assignment, codet &adjusted);
  void static_lifetime_init(const contextt &context, codet &dest);
  virtual void static_lifetime_destroy(codet &dest);

protected:
  bool atexit_may_register() const;
  void add_exit_code(codet &dest, const locationt &location);

  contextt &context;
};

#endif /* CLANG_C_FRONTEND_CLANG_C_MAIN_H_ */
