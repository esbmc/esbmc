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
  virtual bool preprocess(const std::string &path, std::ostream &outstream);

  bool parse(const std::string &path) override;

  bool final(contextt &context) override;

  bool typecheck(contextt &context, const std::string &module) override;

  std::string id() const override
  {
    return "c";
  }

  void show_parse(std::ostream &out) override;

  // conversion from expression into string
  bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns,
    unsigned flags) override;

  // conversion from type into string
  bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns,
    unsigned flags) override;

  unsigned default_flags(presentationt target) const override;

  languaget *new_language() const override
  {
    return new clang_c_languaget();
  }

  clang_c_languaget();

protected:
  virtual std::string internal_additions();

  // Force the file type, .c for the C frontend and .cpp for the C++ one
  virtual void force_file_type(std::vector<std::string> &compiler_args);

  static const std::string &clang_resource_dir();
  virtual void build_compiler_args(std::vector<std::string> &compiler_args);

  std::vector<std::string> compiler_args(std::string tool_name)
  {
    std::vector<std::string> v{std::move(tool_name)};
    force_file_type(v);
    build_compiler_args(v);
    return v;
  }

  std::vector<std::unique_ptr<clang::ASTUnit>> ASTs;
};

languaget *new_clang_c_language();

#endif
