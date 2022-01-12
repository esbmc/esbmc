/*******************************************************************\
Module: Jimple Expr Interface
Author: Rafael Sá Menezes
Date: September 2021
Description: This interface will hold jimple expressions such as
arithmetic, comparation and allocation
\*******************************************************************/

#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_type.h>

#pragma once

/**
 * @brief Base interface to hold jimple expressions
 * 
 */
class jimple_expr : public jimple_ast
{
public:
  virtual exprt
  to_exprt(contextt &, const std::string &, const std::string &) const
  {
    exprt val("at_identifier");
    return val;
  };

  /**
   * @brief Recursively explores and initializes a jimple_expression
   * 
   * Jimple expressions can be unary (e.g. cast) or binary (e.g. addition),
   * however, the operands can be other expressions. For example, it is valid
   * to do a cast over a math operation: `(int) 1 + 1` in Jimple, this should be
   * parsed as:
   * 
   * - cast:
   *   - to: int
   *   - from:
   *     - binop: +
   *       - op1: 1
   *       - op2: 1
   * 
   * This function will do that
   * @param j 
   * @return std::shared_ptr<jimple_expr> 
   */
  static std::shared_ptr<jimple_expr> get_expression(const json &j);
};

/**
 * @brief A number constant (in decimal)
 * 
 * Example: 3, 42, 1, -1, etc...
 * 
 */
class jimple_constant : public jimple_expr
{
public:
  jimple_constant() = default;
  explicit jimple_constant(const std::string &value) : value(value)
  {
  }
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return value;
  }

  virtual void setValue(const std::string &v)
  {
    value = v;
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

/**
 * @brief The value of a symbol (variable)
 * 
 * E.g
 * 
 * int a = 3;
 * 
 * When an expression such as: `1 + a`, it should
 * evaluate to 4
 * 
 */
class jimple_symbol : public jimple_expr
{
public:
  jimple_symbol() = default;
  explicit jimple_symbol(std::string name) : var_name(name)
  {
  }
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

/**
 * @brief A binary operation
 * 
 * E.g. +, -, *, /, ==, !=, etc
 * 
 */
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

/**
 * @brief A cast operation
 * 
 * E.g. (int)
 */
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

/**
 * @brief Get the number of elements of an array (not bits/bytes)
 * 
 * int arr[3];
 * lengthof(arr) = 3
 */
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

/**
 * @brief Allocates a new array (dynamic)
 * 
 * e.g.
 * 
 * int arr[];
 * arr = new int(3);
 */
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

/**
 * @brief The result of a function call
 * 
 * int a = foo();
 * 
 */
class jimple_expr_invoke : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple Invoke";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  const std::string &getBaseClass() const
  {
    return base_class;
  }

  const std::string &getMethod() const
  {
    return method;
  }

  const std::vector<std::shared_ptr<jimple_expr>> &getParameters() const
  {
    return parameters;
  }

  void set_lhs(exprt expr)
  {
    lhs = expr;
  }

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    // TODO: use some hashing to also use the types
    // TODO: DRY
    return std::to_string(parameters.size());
  }
  std::string base_class;
  std::string method;
  exprt lhs;
  std::vector<std::shared_ptr<jimple_expr>> parameters;
};

/**
 * @brief Allocates a type
 * 
 * Note: this is not a constructor! The constructor is called through
 * an invoke_stmt.
 * 
 * Example: 
 * 
 * CustomClass c;
 * c = new CustomClass()
 * 
 * v v v v v
 * 
 * CustomClass *c;
 * c = alloca(sizeof(CustomClass)) // JIMPLE_NEW
 * c.<init>() // the inner constructor
 * 
 */
class jimple_new : public jimple_newarray
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple New";
  }
};

/**
 * @brief Access of arrays
 * 
 * Since in Jimple arrays are treated as pointers
 * this is a pointer arithmetic expression
 * 
 * arr[4] // JIMPLE_DEREF
 * 
 */
class jimple_deref : public jimple_expr
{
public:
  jimple_deref() = default;
  jimple_deref(
    std::shared_ptr<jimple_expr> index,
    std::shared_ptr<jimple_expr> base)
    : index(index), base(base)
  {
  }
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

/**
 * @brief Nondet call
 * 
 * // TODO: implement this
 * 
 */
class jimple_nondet : public jimple_expr
{
public:
  jimple_nondet() = default;
  virtual std::string to_string() const override
  {
    return "Jimple Nondet";
  }
  virtual void from_json(const json &) override
  {
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
};

/**
 * @brief A field member of object
 * 
 * class Foo { int a; }
 * 
 * Foo f = ...;
 * int a = F.a; // F.a is a field access
 * 
 */
class jimple_field_access : public jimple_expr
{
public:
  virtual std::string to_string() const override
  {
    return "Jimple Field Access";
  }
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

protected:
  std::string from;
  std::string field;
  std::shared_ptr<jimple_type> type;
};