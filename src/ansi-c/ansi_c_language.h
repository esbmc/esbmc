/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_LANGUAGE_H
#define CPROVER_ANSI_C_LANGUAGE_H

#include <ansi-c/ansi_c_parse_tree.h>
#include <util/language.h>

class ansi_c_languaget : public languaget
{
public:
  virtual bool preprocess(
    const std::string &path,
    std::ostream &outstream,
    const messaget &message_handler);

  bool parse(const std::string &path, const messaget &message_handler) override;

  bool typecheck(
    contextt &context,
    const std::string &module,
    const messaget &message_handler) override;

  bool final(contextt &context, const messaget &message_handler) override;

  virtual bool merge_context(
    contextt &dest,
    contextt &src,
    const messaget &message_handler,
    const std::string &module) const;

  void show_parse(std::ostream &out) override;

  ~ansi_c_languaget() override = default;
  explicit ansi_c_languaget(const messaget &msg);

  // conversion from expression into string
  bool from_expr(const exprt &expr, std::string &code, const namespacet &ns)
    override;

  // conversion from type into string
  bool from_type(const typet &type, std::string &code, const namespacet &ns)
    override;

  languaget *new_language(const messaget &msg) const override
  {
    return new ansi_c_languaget(msg);
  }

  std::string id() const override
  {
    return "C";
  }

protected:
  ansi_c_parse_treet parse_tree;
  std::string parse_path;
};

languaget *new_ansi_c_language(const messaget &msg);

#endif
