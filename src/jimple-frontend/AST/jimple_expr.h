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
class jimple_lengthof : public jimple_expr
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

  std::string base_class;
  std::string method;
  exprt lhs;
  std::vector<std::shared_ptr<jimple_expr>> parameters;

  void set_lhs(exprt expr)
  {
    lhs = expr;
  }

  bool is_nondet_call() const
  {
    return base_class == "org.sosy_lab.sv_benchmarks.Verifier";
  }

  bool is_intrinsic_method = false;

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    return std::to_string(parameters.size());
  }
};

/**
 * @brief The result of a function call
 *
 * int a = foo();
 *
 */
class jimple_virtual_invoke : public jimple_expr
{
public:
  virtual void from_json(const json &j) override;
  virtual std::string to_string() const override
  {
    return "Jimple Virtual Invoke";
  }

  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::string base_class;
  std::string method;
  exprt lhs;
  std::string variable;
  std::vector<std::shared_ptr<jimple_expr>> parameters;

  void set_lhs(exprt expr)
  {
    lhs = expr;
  }

  bool is_nondet_call() const
  {
    return base_class == "java.util.Random";
  }

protected:
  // We need an unique name for each function
  std::string get_hash_name() const
  {
    return std::to_string(parameters.size() + 1);
  }
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

  std::shared_ptr<jimple_expr> index;
  std::shared_ptr<jimple_expr> base;
};

/**
 * @brief Nondet call (This in an extension)
 *
 */
class jimple_nondet : public jimple_expr
{
public:
  jimple_nondet() = default;
  explicit jimple_nondet(std::string mode) : mode(mode)
  {
  }
  virtual std::string to_string() const override
  {
    return "Jimple Nondet";
  }
  virtual void from_json(const json &) override
  {
    assert("This class shouldn't be used from Jimple directly");
    abort();
  }

  const std::string mode; // Int, char, long, etc... e.g. Random().nextInt()
  //std::string bound;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;
};

/**
 * @brief A static member of object/class
 *
 * class Foo { static int a; }
 *
 * Foo f = ...;
 * int a = F.a; // F.a (or Foo.a) is a static member access
 *
 */
class jimple_static_member : public jimple_expr
{
public:
  virtual std::string to_string() const override
  {
    return "Jimple Static Member";
  }
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::string from;
  std::string field;
  std::shared_ptr<jimple_type> type;
};

/**
 * @brief A virtual member of object
 *
 * class Foo { int a; }
 *
 * Foo f = ...;
 * int a = F.a; // F.a is a virtual member access
 *
 */
class jimple_virtual_member : public jimple_expr
{
public:
  virtual std::string to_string() const override
  {
    return "Jimple Virtual Member";
  }
  virtual void from_json(const json &j) override;
  virtual exprt to_exprt(
    contextt &ctx,
    const std::string &class_name,
    const std::string &function_name) const override;

  std::string variable;
  std::string from;
  std::string field;
  std::shared_ptr<jimple_type> type;
};
