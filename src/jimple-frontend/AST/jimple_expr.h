#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_type.h>

#pragma once
class jimple_expr : public jimple_ast
{
public:
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const
  {
    exprt val("at_identifier");
    return val;
  };

  // Expressions parsing can be recursive
  static std::shared_ptr<jimple_expr> get_expression(const json &j);
};

class jimple_constant : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return value;
  }
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
  
  const std::string &getValue() const
  {
    return value;
  }

protected:
  std::string value;
};

class jimple_symbol : public jimple_expr
{
public:
  jimple_symbol() = default;
  explicit jimple_symbol(std::string name) : var_name(name) {}
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return var_name;
  }
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getVarName() const
  {
    return var_name;
  }

protected:
  std::string var_name;
};

class jimple_binop : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple BinOP";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getBinop() const
  {
    return binop;
  }

  std::shared_ptr<jimple_expr> &getLhs()
  {
    return lhs;
  }

  std::shared_ptr<jimple_expr> &getRhs()
  {
    return rhs;
  }

protected:
  std::string binop;
  std::shared_ptr<jimple_expr> lhs;
  std::shared_ptr<jimple_expr> rhs;
};

class jimple_cast : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple Cast";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getVarName() const
  {
    return var_name;
  }

  std::shared_ptr<jimple_expr> &getFrom()
  {
    return from;
  }

  std::shared_ptr<jimple_type> &getType()
  {
    return to;
  }

protected:
  std::string var_name;
  std::shared_ptr<jimple_expr> from;
  std::shared_ptr<jimple_type> to;
};

class jimple_lenghof : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple Lengthof";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::shared_ptr<jimple_expr> &getFrom()
  {
    return from;
  }

protected:
  std::shared_ptr<jimple_expr> from;
};

class jimple_newarray : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple New array";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::shared_ptr<jimple_expr> &getSize()
  {
    return size;
  }


  std::shared_ptr<jimple_type> &getType()
  {
    return type;
  }

protected:
  std::shared_ptr<jimple_type> type;
  std::shared_ptr<jimple_expr> size;
};

class jimple_deref : public jimple_expr
{
public:
  jimple_deref() = default;
  jimple_deref(std::shared_ptr<jimple_expr> index,  std::shared_ptr<jimple_expr> base ): index(index), base(base) {}
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple Deref";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::shared_ptr<jimple_expr> &getIndex()
  {
    return index;
  }


  std::shared_ptr<jimple_expr> &getBase()
  {
    return base;
  }

protected:
  std::shared_ptr<jimple_expr> index;
  std::shared_ptr<jimple_expr> base;
};


class jimple_nondet : public jimple_expr
{
public:
  jimple_nondet() = default;  
  virtual std::string to_string() const override
  {
    return "Jimple Nondet";
  }
  virtual void from_json(const json &j) override {}

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

};