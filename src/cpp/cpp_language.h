/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_LANGUAGE_H
#define CPROVER_CPP_LANGUAGE_H

#include <cpp/cpp_parse_tree.h>
#include <util/language.h>

class cpp_languaget : public languaget
{
public:
  virtual bool preprocess(const std::string &path, std::ostream &outstream);

  bool parse(const std::string &path) override;

  bool typecheck(contextt &context, const std::string &module) override;

  bool merge_context(
    contextt &dest,
    contextt &src,
    const std::string &module,
    class replace_symbolt &replace_symbol) const;

  bool final(contextt &context) override;

  void show_parse(std::ostream &out) override;

  // constructor, destructor
  ~cpp_languaget() override = default;
  explicit cpp_languaget();

  // conversion from expression into string
  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  // conversion from type into string
  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

  languaget *new_language() const override
  {
    return new cpp_languaget();
  }

protected:
  cpp_parse_treet cpp_parse_tree;
  std::string parse_path;

  void show_parse(std::ostream &out, const cpp_itemt &item);
  void internal_additions(std::ostream &outstream);

  virtual std::string main_symbol()
  {
    return "main";
  }
};

languaget *new_cpp_language();

#endif
