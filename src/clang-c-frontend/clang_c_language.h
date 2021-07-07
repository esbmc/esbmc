/*******************************************************************\

Module: Clang C Language Module

Author:

\*******************************************************************/

#ifndef CLANG_C_FRONTEND_CLANG_C_LANGUAGE_H_
#define CLANG_C_FRONTEND_CLANG_C_LANGUAGE_H_

#include <util/language.h>

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

// Forward dec, to avoid bringing in clang headers
namespace clang
{
class ASTUnit;
} // namespace clang

class clang_c_languaget : public languaget
{
public:
  virtual bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    const messaget &msg);

  bool parse(const std::string &path, const messaget &msg) override;

  bool final(contextt &context, const messaget &msg) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    const messaget &msg) override;

  std::string id() const override
  {
    return "c";
  }

  void show_parse(std::ostream &out) override;

  // conversion from expression into string
  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  // conversion from type into string
  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

  languaget *new_language(const messaget &msg) override
  {
    return new clang_c_languaget(msg);
  }

  // constructor, destructor
  ~clang_c_languaget() override = default;
  explicit clang_c_languaget(const messaget &msg);

protected:
  virtual std::string internal_additions();
  virtual void force_file_type();

  void dump_clang_headers(const std::string &tmp_dir);
  void build_compiler_args(const std::string &&tmp_dir);

  std::vector<std::string> compiler_args;
  std::vector<std::unique_ptr<clang::ASTUnit>> ASTs;
};

languaget *new_clang_c_language(const messaget &msg);

#endif
