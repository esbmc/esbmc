#pragma once

#include <util/context.h>
#include <cstddef>
#include <deque>
#include <iosfwd>
#include <string>
#include <string_view>

class expression_node
{
public:
  enum class node_kind
  {
    unknown,
    identifier,
    integer_literal,

    unary_plus,
    unary_minus,
    logical_not,

    add,
    subtract,
    multiply,
    divide,

    less,
    less_equal,
    greater,
    greater_equal,
    equal,
    not_equal,

    logical_and,
    logical_or,

    member,
    index
  };

public:
  explicit expression_node(
    std::string_view node_value = std::string_view(),
    node_kind node_type = node_kind::unknown);

  // Node text: operator, identifier, or integer literal.
  std::string_view value;

  // Semantic node kind for later AST conversion.
  node_kind kind;

  // Unary nodes use only left.
  // Binary nodes use both left and right.
  const expression_node *left;
  const expression_node *right;

  // Write the AST to standard output.
  void dump() const;

  // Write the AST to the given output stream.
  void output(std::ostream &out) const;

private:
  static const char *kind_to_string(node_kind kind);

  // Recursive helper for tree printing.
  static void node_output(
    const expression_node *node,
    std::ostream &out,
    const std::string &prefix,
    bool has_next_sibling);
};

class expression_parser
{
public:
  expression_parser();

  // Parse an expression and return the AST root through `root`.
  // Returns true on error, false on success.
  // The input string must remain valid while the returned AST is used.
  bool parse(std::string_view expression, const expression_node *&root);

private:
  enum class token_type
  {
    end,
    identifier,
    integer,
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    dot,
    arrow,
    plus,
    minus,
    star,
    slash,
    less,
    less_equal,
    greater,
    greater_equal,
    equal_equal,
    not_equal,
    logical_not,
    logical_and,
    logical_or
  };

  struct token
  {
    token_type type;
    std::size_t begin;
    std::size_t length;
  };

private:
  // Parse using precedence climbing.
  bool parse_expression(int minimum_precedence, const expression_node *&node);

  // Parse unary prefix operators.
  bool parse_prefix(const expression_node *&node);

  // Parse postfix operators.
  bool parse_postfix(const expression_node *&node);

  // Parse identifiers, integers, and parenthesized expressions.
  bool parse_primary(const expression_node *&node);

  // Read the next token from the input.
  // Returns true on lexical error, false otherwise.
  bool next_token();

  // Return precedence for a binary operator.
  int binary_precedence(token_type type) const;

  // Map a binary token to an AST node kind.
  expression_node::node_kind binary_node_kind(token_type type) const;

  // Return the original source text for a token.
  std::string_view token_text(const token &tok) const;

  // Allocate a node from the internal pool.
  const expression_node *make_node(
    std::string_view value,
    expression_node::node_kind kind,
    const expression_node *left = nullptr,
    const expression_node *right = nullptr);

private:
  std::string_view input_;
  std::size_t position_;
  token current_;
  std::deque<expression_node> nodes_;
};

class expression_converter
{
public:
  explicit expression_converter(contextt &ns, locationt &location);
  // Entry point
  bool convert(const expression_node *root, exprt &expr);
  bool get_expr(const expression_node &node, exprt &expr);

private:
  contextt &context_;
  locationt &location_;
};
