#pragma once

#include <cstddef>
#include <deque>
#include <iosfwd>
#include <string_view>

class expression_node
{
public:
  explicit expression_node(std::string_view node_value = std::string_view());

  // Node text: operator, identifier, or integer literal.
  std::string_view value;

  // Unary nodes use only left.
  // Binary nodes use both left and right.
  const expression_node *left;
  const expression_node *right;

  // Print the AST as a tree.
  void print(std::ostream &output_stream) const;
  void print() const;

private:
  // Recursive helper for tree printing.
  static void print_impl(
    const expression_node *node,
    std::ostream &output_stream,
    const std::string &prefix,
    bool has_next_sibling);
};

class expression_parser
{
public:
  expression_parser();

  // Parse an expression and return the AST root.
  // The input string must remain valid while the returned AST is used.
  const expression_node *parse(std::string_view expression);

private:
  enum class token_type
  {
    end,
    identifier,
    integer,
    left_paren,
    right_paren,
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
  const expression_node *parse_expression(int minimum_precedence);

  // Parse unary prefix operators.
  const expression_node *parse_prefix();

  // Parse identifiers, integers, and parenthesized expressions.
  const expression_node *parse_primary();

  // Read the next token from the input.
  void next_token();

  // Return precedence for a binary operator.
  int binary_precedence(token_type type) const;

  // Return the original source text for a token.
  std::string_view token_text(const token &tok) const;

  // Allocate a node from the internal pool.
  const expression_node *make_node(
    std::string_view value,
    const expression_node *left = nullptr,
    const expression_node *right = nullptr);

  // Raise a parse error with source position.
  [[noreturn]] void throw_error(const char *message, std::size_t position) const;

private:
  std::string_view input_;
  std::size_t position_;
  token current_;
  std::deque<expression_node> nodes_;
};
