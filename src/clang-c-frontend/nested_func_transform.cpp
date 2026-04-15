#include <clang-c-frontend/nested_func_transform.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// A local variable declaration found in an enclosing function scope.
struct local_var
{
  std::string type_text; // e.g. "int", "struct foo *"
  std::string name;
};

// A nested function extracted from its enclosing function.
struct nested_func
{
  std::string return_type;         // verbatim return-type text
  std::string name;                // original identifier
  std::string params_text;         // text inside () including parens
  std::string body_text;           // text from '{' to matching '}' inclusive
  std::string enclosing;           // enclosing function name
  size_t def_start;                // byte offset of the start of the definition
  size_t def_end;                  // byte offset one past the closing '}'
  bool used_as_fptr;               // true if name appears in non-call context
  std::vector<local_var> captures; // variables captured from enclosing scope
};

// -----------------------------------------------------------------------
//  Lightweight C token scanner
// -----------------------------------------------------------------------

enum class tok_kind
{
  identifier,
  punctuation,
  number,
  string_lit,
  char_lit,
  pp_directive,
  whitespace,
  eof
};

struct token
{
  tok_kind kind;
  std::string text;
  size_t pos; // byte offset in source
};

static bool is_ident_start(char c)
{
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool is_ident_cont(char c)
{
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Tokenize `src` into a flat vector.  Comments are skipped (replaced by
// a single whitespace token).  String/char literals are kept as opaque
// blobs.  Preprocessor directives are kept as single tokens.
static std::vector<token> tokenize(const std::string &src)
{
  std::vector<token> tokens;
  size_t i = 0;
  const size_t n = src.size();

  while (i < n)
  {
    // --- whitespace ---
    if (std::isspace(static_cast<unsigned char>(src[i])))
    {
      size_t start = i;
      while (i < n && std::isspace(static_cast<unsigned char>(src[i])))
        ++i;
      tokens.push_back(
        {tok_kind::whitespace, src.substr(start, i - start), start});
      continue;
    }

    // --- line comment ---
    if (i + 1 < n && src[i] == '/' && src[i + 1] == '/')
    {
      size_t start = i;
      while (i < n && src[i] != '\n')
        ++i;
      tokens.push_back(
        {tok_kind::whitespace, src.substr(start, i - start), start});
      continue;
    }

    // --- block comment ---
    if (i + 1 < n && src[i] == '/' && src[i + 1] == '*')
    {
      size_t start = i;
      i += 2;
      while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/'))
        ++i;
      if (i + 1 < n)
        i += 2;
      tokens.push_back(
        {tok_kind::whitespace, src.substr(start, i - start), start});
      continue;
    }

    // --- preprocessor directive (# at the beginning of a line) ---
    if (src[i] == '#')
    {
      // check it's at line start (only whitespace before on this line)
      bool at_line_start = true;
      for (size_t j = i; j > 0; --j)
      {
        char c = src[j - 1];
        if (c == '\n')
          break;
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
          at_line_start = false;
          break;
        }
      }
      if (at_line_start)
      {
        size_t start = i;
        while (i < n)
        {
          if (src[i] == '\n')
          {
            // check for line continuation
            if (i > 0 && src[i - 1] == '\\')
            {
              ++i;
              continue;
            }
            ++i;
            break;
          }
          ++i;
        }
        tokens.push_back(
          {tok_kind::pp_directive, src.substr(start, i - start), start});
        continue;
      }
    }

    // --- string literal ---
    if (src[i] == '"')
    {
      size_t start = i;
      ++i;
      while (i < n && src[i] != '"')
      {
        if (src[i] == '\\' && i + 1 < n)
          ++i; // skip escaped char
        ++i;
      }
      if (i < n)
        ++i; // consume closing quote
      tokens.push_back(
        {tok_kind::string_lit, src.substr(start, i - start), start});
      continue;
    }

    // --- character literal ---
    if (src[i] == '\'')
    {
      size_t start = i;
      ++i;
      while (i < n && src[i] != '\'')
      {
        if (src[i] == '\\' && i + 1 < n)
          ++i;
        ++i;
      }
      if (i < n)
        ++i;
      tokens.push_back(
        {tok_kind::char_lit, src.substr(start, i - start), start});
      continue;
    }

    // --- identifier / keyword ---
    if (is_ident_start(src[i]))
    {
      size_t start = i;
      while (i < n && is_ident_cont(src[i]))
        ++i;
      tokens.push_back(
        {tok_kind::identifier, src.substr(start, i - start), start});
      continue;
    }

    // --- number ---
    if (std::isdigit(static_cast<unsigned char>(src[i])))
    {
      size_t start = i;
      while (i < n && (is_ident_cont(src[i]) || src[i] == '.'))
        ++i;
      // Handle suffixes like 0x, exponents
      tokens.push_back({tok_kind::number, src.substr(start, i - start), start});
      continue;
    }

    // --- punctuation (single char) ---
    {
      tokens.push_back({tok_kind::punctuation, std::string(1, src[i]), i});
      ++i;
    }
  }

  tokens.push_back({tok_kind::eof, "", n});
  return tokens;
}

// -----------------------------------------------------------------------
//  Helper: skip whitespace/pp tokens in the token stream
// -----------------------------------------------------------------------

static size_t skip_ws(const std::vector<token> &toks, size_t idx)
{
  while (idx < toks.size() && (toks[idx].kind == tok_kind::whitespace ||
                               toks[idx].kind == tok_kind::pp_directive))
    ++idx;
  return idx;
}

// -----------------------------------------------------------------------
//  Type keyword recognition
// -----------------------------------------------------------------------

static const std::set<std::string> &type_keywords()
{
  static const std::set<std::string> kw = {
    "void",
    "int",
    "char",
    "short",
    "long",
    "float",
    "double",
    "unsigned",
    "signed",
    "struct",
    "union",
    "enum",
    "const",
    "volatile",
    "static",
    "inline",
    "_Bool",
    "__auto_type"};
  return kw;
}

static bool is_type_keyword(const std::string &s)
{
  return type_keywords().count(s) > 0;
}

// C keywords that can appear before '(' but are NOT function names.
static bool is_non_func_keyword(const std::string &s)
{
  static const std::set<std::string> kw = {
    "if",
    "else",
    "while",
    "for",
    "do",
    "switch",
    "case",
    "default",
    "return",
    "break",
    "continue",
    "goto",
    "sizeof",
    "typeof",
    "alignof",
    "_Alignof",
    "_Generic",
    "_Static_assert",
    "__typeof",
    "__typeof__",
    "__alignof",
    "__alignof__",
    "__attribute__",
    "__attribute",
    "__asm__",
    "__asm",
    "asm",
    "typedef",
    "extern",
    "register",
    "auto",
    "__extension__",
    "__builtin_va_arg",
    "__builtin_offsetof",
    "__builtin_types_compatible_p"};
  return kw.count(s) > 0;
}

// -----------------------------------------------------------------------
//  Detect and extract nested functions
// -----------------------------------------------------------------------

// Try to parse a function definition starting at toks[idx].
// Returns true if successful, and fills the nested_func struct.
// `enclosing_name` is the name of the function we're currently inside.
// Only call this when brace_depth >= 1 (inside a function body).
static bool try_parse_func_def(
  const std::vector<token> &toks,
  size_t idx,
  const std::string &enclosing_name,
  nested_func &out,
  size_t &def_end_tok_idx,
  const std::string &src)
{
  // We need: [type-tokens]+ IDENTIFIER ( ... ) {
  // type-tokens are type keywords, identifiers (typedef names), or '*'
  size_t start_pos = toks[idx].pos;

  // Collect return type tokens
  std::string return_type;
  bool has_type_keyword = false;

  while (idx < toks.size())
  {
    size_t next = skip_ws(toks, idx);
    if (next >= toks.size())
      return false;

    const token &t = toks[next];

    if (t.kind == tok_kind::identifier && is_type_keyword(t.text))
    {
      if (!return_type.empty())
        return_type += " ";
      return_type += t.text;
      has_type_keyword = true;
      idx = next + 1;

      // struct/union/enum are followed by a tag name
      if (t.text == "struct" || t.text == "union" || t.text == "enum")
      {
        size_t tag = skip_ws(toks, idx);
        if (tag < toks.size() && toks[tag].kind == tok_kind::identifier)
        {
          return_type += " " + toks[tag].text;
          idx = tag + 1;
        }
      }
      continue;
    }

    // pointer stars
    if (t.kind == tok_kind::punctuation && t.text == "*")
    {
      return_type += " *";
      idx = next + 1;
      continue;
    }

    // An identifier that might be a typedef name (not a type keyword)
    // Only accept this as type if we haven't seen any type keywords yet
    // and the next meaningful token after it and its name and ( is {
    if (t.kind == tok_kind::identifier && !has_type_keyword)
    {
      // Peek ahead: is this IDENT IDENT ( ... ) { ?
      size_t peek = skip_ws(toks, next + 1);
      if (peek < toks.size() && toks[peek].kind == tok_kind::identifier)
      {
        size_t peek2 = skip_ws(toks, peek + 1);
        if (
          peek2 < toks.size() && toks[peek2].kind == tok_kind::punctuation &&
          toks[peek2].text == "(")
        {
          return_type = t.text;
          has_type_keyword = true; // treat typedef name as type
          idx = next + 1;
          continue;
        }
      }
    }

    break;
  }

  if (return_type.empty())
    return false;

  // Now expect: IDENTIFIER
  size_t name_idx = skip_ws(toks, idx);
  if (name_idx >= toks.size() || toks[name_idx].kind != tok_kind::identifier)
    return false;

  // Make sure this isn't a type keyword or a C control-flow keyword
  if (is_type_keyword(toks[name_idx].text))
    return false;
  if (is_non_func_keyword(toks[name_idx].text))
    return false;

  std::string func_name = toks[name_idx].text;
  idx = name_idx + 1;

  // Now expect: (
  size_t paren_idx = skip_ws(toks, idx);
  if (
    paren_idx >= toks.size() || toks[paren_idx].kind != tok_kind::punctuation ||
    toks[paren_idx].text != "(")
    return false;

  // Find matching )
  int paren_depth = 1;
  size_t params_start = toks[paren_idx].pos;
  idx = paren_idx + 1;
  while (idx < toks.size() && paren_depth > 0)
  {
    if (toks[idx].kind == tok_kind::punctuation)
    {
      if (toks[idx].text == "(")
        ++paren_depth;
      else if (toks[idx].text == ")")
        --paren_depth;
    }
    ++idx;
  }
  if (paren_depth != 0)
    return false;
  size_t params_end = toks[idx - 1].pos + 1; // one past ')'

  // Now expect: {
  size_t brace_idx = skip_ws(toks, idx);
  if (
    brace_idx >= toks.size() || toks[brace_idx].kind != tok_kind::punctuation ||
    toks[brace_idx].text != "{")
    return false;

  // Find matching }
  int brace_depth = 1;
  size_t body_start = toks[brace_idx].pos;
  idx = brace_idx + 1;
  while (idx < toks.size() && brace_depth > 0)
  {
    if (toks[idx].kind == tok_kind::punctuation)
    {
      if (toks[idx].text == "{")
        ++brace_depth;
      else if (toks[idx].text == "}")
        --brace_depth;
    }
    ++idx;
  }
  if (brace_depth != 0)
    return false;
  size_t body_end = toks[idx - 1].pos + 1; // one past '}'

  out.return_type = return_type;
  out.name = func_name;
  out.params_text = src.substr(params_start, params_end - params_start);
  out.body_text = src.substr(body_start, body_end - body_start);
  out.enclosing = enclosing_name;
  out.def_start = start_pos;
  out.def_end = body_end;
  out.used_as_fptr = false;

  def_end_tok_idx = idx;
  return true;
}

// Collect local variable declarations from a function body.
// This is a best-effort heuristic that handles common patterns:
//   type [*]* name [= ...] ;
//   type [*]* name , name2 ;
// Also includes function parameters.
static std::vector<local_var> collect_local_vars(
  const std::vector<token> &toks,
  size_t body_start_tok,
  size_t body_end_tok,
  const std::set<std::string> &nested_func_names)
{
  std::vector<local_var> vars;
  size_t idx = body_start_tok;

  // We only look at depth 1 (the direct function body, not nested blocks)
  int depth = 0;

  while (idx < body_end_tok)
  {
    if (toks[idx].kind == tok_kind::punctuation)
    {
      if (toks[idx].text == "{")
        ++depth;
      else if (toks[idx].text == "}")
        --depth;
    }

    // Only parse declarations at the direct function body level (depth 1).
    // Skip deeper nesting (depth > 1) to avoid collecting variables from
    // nested function definitions or inner blocks.
    if (depth == 1 && toks[idx].kind == tok_kind::identifier)
    {
      // Try: type-keyword(s) [*]* name [= ...] [, name2 ...] ;
      size_t try_idx = idx;
      std::string type_str;
      bool found_type = false;

      // Collect type tokens
      while (try_idx < body_end_tok)
      {
        size_t next = skip_ws(toks, try_idx);
        if (next >= body_end_tok)
          break;

        const token &t = toks[next];
        if (t.kind == tok_kind::identifier && is_type_keyword(t.text))
        {
          if (!type_str.empty())
            type_str += " ";
          type_str += t.text;
          found_type = true;
          try_idx = next + 1;

          if (t.text == "struct" || t.text == "union" || t.text == "enum")
          {
            size_t tag = skip_ws(toks, try_idx);
            if (tag < body_end_tok && toks[tag].kind == tok_kind::identifier)
            {
              type_str += " " + toks[tag].text;
              try_idx = tag + 1;
            }
          }
          continue;
        }

        if (t.kind == tok_kind::punctuation && t.text == "*" && found_type)
        {
          type_str += " *";
          try_idx = next + 1;
          continue;
        }

        break;
      }

      if (!found_type)
      {
        // Typedef heuristic: IDENT IDENT followed by = ; [ or ,
        // e.g. "aligned jj;" where aligned is a typedef
        size_t t1 = skip_ws(toks, idx);
        if (
          t1 < body_end_tok && toks[t1].kind == tok_kind::identifier &&
          !is_type_keyword(toks[t1].text) &&
          !is_non_func_keyword(toks[t1].text))
        {
          size_t t2 = skip_ws(toks, t1 + 1);
          if (
            t2 < body_end_tok && toks[t2].kind == tok_kind::identifier &&
            !is_type_keyword(toks[t2].text) &&
            !is_non_func_keyword(toks[t2].text) &&
            !nested_func_names.count(toks[t2].text))
          {
            size_t t3 = skip_ws(toks, t2 + 1);
            if (
              t3 < body_end_tok && toks[t3].kind == tok_kind::punctuation &&
              (toks[t3].text == "=" || toks[t3].text == ";" ||
               toks[t3].text == "[" || toks[t3].text == ","))
            {
              vars.push_back({toks[t1].text, toks[t2].text});
              // Skip past this declaration
              size_t scan = t3;
              while (scan < body_end_tok)
              {
                if (
                  toks[scan].kind == tok_kind::punctuation &&
                  toks[scan].text == ";")
                {
                  idx = scan + 1;
                  break;
                }
                ++scan;
              }
              if (scan >= body_end_tok)
                idx = scan;
              continue;
            }
          }
        }
        ++idx;
        continue;
      }

      // Now expect one or more: name [= expr] [, ...]  ;
      while (true)
      {
        size_t name_idx = skip_ws(toks, try_idx);
        if (
          name_idx >= body_end_tok ||
          toks[name_idx].kind != tok_kind::identifier ||
          is_type_keyword(toks[name_idx].text))
          break;

        std::string var_name = toks[name_idx].text;

        // Don't treat nested function names as variables.
        // Skip the entire nested function definition (params + body).
        if (nested_func_names.count(var_name))
        {
          size_t skip = skip_ws(toks, name_idx + 1);
          if (
            skip < body_end_tok && toks[skip].kind == tok_kind::punctuation &&
            toks[skip].text == "(")
          {
            // Skip past matching )
            int pd = 1;
            ++skip;
            while (skip < body_end_tok && pd > 0)
            {
              if (toks[skip].kind == tok_kind::punctuation)
              {
                if (toks[skip].text == "(")
                  ++pd;
                else if (toks[skip].text == ")")
                  --pd;
              }
              ++skip;
            }
            // Skip past matching }
            size_t brace = skip_ws(toks, skip);
            if (
              brace < body_end_tok &&
              toks[brace].kind == tok_kind::punctuation &&
              toks[brace].text == "{")
            {
              int bd = 1;
              ++brace;
              while (brace < body_end_tok && bd > 0)
              {
                if (toks[brace].kind == tok_kind::punctuation)
                {
                  if (toks[brace].text == "{")
                    ++bd;
                  else if (toks[brace].text == "}")
                    --bd;
                }
                ++brace;
              }
              try_idx = brace;
              goto done_with_decl;
            }
          }
          break;
        }

        // Check what follows: = or , or ; means it's a declaration
        size_t after = skip_ws(toks, name_idx + 1);
        if (after >= body_end_tok)
          break;

        bool is_var = false;
        if (
          toks[after].kind == tok_kind::punctuation &&
          (toks[after].text == "=" || toks[after].text == "," ||
           toks[after].text == ";"))
        {
          is_var = true;
        }
        // Also handle: name [ ... ] = ... (array)
        if (
          toks[after].kind == tok_kind::punctuation && toks[after].text == "[")
        {
          is_var = true;
        }

        if (is_var)
        {
          vars.push_back({type_str, var_name});

          // Skip to , or ;
          size_t scan = after;
          int skip_depth = 0;
          while (scan < body_end_tok)
          {
            if (toks[scan].kind == tok_kind::punctuation)
            {
              if (
                toks[scan].text == "(" || toks[scan].text == "[" ||
                toks[scan].text == "{")
                ++skip_depth;
              else if (
                toks[scan].text == ")" || toks[scan].text == "]" ||
                toks[scan].text == "}")
                --skip_depth;
              else if (skip_depth == 0 && toks[scan].text == ",")
              {
                try_idx = scan + 1;
                break;
              }
              else if (skip_depth == 0 && toks[scan].text == ";")
              {
                try_idx = scan + 1;
                goto done_with_decl;
              }
            }
            ++scan;
          }
          continue;
        }
        break;
      }

    done_with_decl:
      idx = try_idx;
      continue;
    }

    ++idx;
  }

  return vars;
}

// Collect parameter names from a parameter list like "(int a, int b)"
static std::vector<local_var> collect_params(const std::string &params_text)
{
  std::vector<local_var> vars;
  auto toks = tokenize(params_text);

  // Skip outer parentheses
  if (toks.size() < 2)
    return vars;

  // Split by commas at paren depth 1, extract last identifier before each
  // comma/end that is NOT inside brackets (to handle VLA params like int t[b]).
  int depth = 0;
  int bracket_depth = 0;
  std::string current_type;
  std::string last_ident;
  std::vector<std::string> type_parts;

  for (size_t i = 0; i < toks.size(); ++i)
  {
    const token &t = toks[i];
    if (t.kind == tok_kind::whitespace || t.kind == tok_kind::eof)
      continue;

    if (t.kind == tok_kind::punctuation)
    {
      if (t.text == "(")
      {
        ++depth;
        continue;
      }
      if (t.text == ")")
      {
        --depth;
        if (depth == 0 && !last_ident.empty())
        {
          std::string type;
          for (const auto &tp : type_parts)
          {
            if (!type.empty())
              type += " ";
            type += tp;
          }
          vars.push_back({type, last_ident});
        }
        continue;
      }
      if (t.text == "[" && depth == 1)
      {
        ++bracket_depth;
        continue;
      }
      if (t.text == "]" && depth == 1)
      {
        --bracket_depth;
        continue;
      }
      if (t.text == "," && depth == 1 && bracket_depth == 0)
      {
        if (!last_ident.empty())
        {
          std::string type;
          for (const auto &tp : type_parts)
          {
            if (!type.empty())
              type += " ";
            type += tp;
          }
          vars.push_back({type, last_ident});
        }
        last_ident.clear();
        type_parts.clear();
        continue;
      }
      if (t.text == "*" && depth == 1 && bracket_depth == 0)
      {
        if (!last_ident.empty())
        {
          type_parts.push_back(last_ident);
          last_ident.clear();
        }
        type_parts.push_back("*");
        continue;
      }
    }

    // Only treat identifiers outside brackets as potential param names
    if (t.kind == tok_kind::identifier && depth == 1 && bracket_depth == 0)
    {
      if (!last_ident.empty())
        type_parts.push_back(last_ident);
      last_ident = t.text;
    }
  }

  return vars;
}

// Check if the nested function's name is used in a non-call context
// (i.e., as a function pointer value).
static bool check_fptr_use(
  const std::vector<token> &toks,
  size_t body_start_tok,
  size_t body_end_tok,
  size_t def_start_tok,
  size_t def_end_tok,
  const std::string &func_name)
{
  for (size_t i = body_start_tok; i < body_end_tok; ++i)
  {
    // Skip the nested function definition itself
    if (i >= def_start_tok && i < def_end_tok)
      continue;

    if (toks[i].kind == tok_kind::identifier && toks[i].text == func_name)
    {
      // Check what follows: if it's '(' then it's a direct call
      size_t next = skip_ws(toks, i + 1);
      if (
        next < toks.size() && toks[next].kind == tok_kind::punctuation &&
        toks[next].text == "(")
      {
        continue; // direct call, not fptr use
      }
      return true; // used as value
    }
  }
  return false;
}

// Identify which enclosing variables are captured by a nested function
static std::vector<local_var> find_captures(
  const std::string &body_text,
  const std::vector<local_var> &enclosing_vars,
  const std::string &params_text)
{
  auto body_toks = tokenize(body_text);
  auto param_vars = collect_params(params_text);

  // Build set of parameter names (these shadow enclosing vars)
  std::set<std::string> param_names;
  for (const auto &p : param_vars)
    param_names.insert(p.name);

  // Build set of local variable names declared inside the body
  std::set<std::string> no_nested;
  auto locals = collect_local_vars(body_toks, 0, body_toks.size(), no_nested);
  std::set<std::string> local_names;
  for (const auto &l : locals)
    local_names.insert(l.name);

  std::vector<local_var> captures;
  std::set<std::string> captured_names;

  // Also tokenize params to find captured vars in VLA expressions
  auto params_toks = tokenize(params_text);

  for (const auto &ev : enclosing_vars)
  {
    if (param_names.count(ev.name) || local_names.count(ev.name))
      continue; // shadowed

    if (captured_names.count(ev.name))
      continue;

    // Check if this identifier appears in the body or param types
    bool found = false;
    for (const auto &t : body_toks)
    {
      if (t.kind == tok_kind::identifier && t.text == ev.name)
      {
        found = true;
        break;
      }
    }
    if (!found)
    {
      for (const auto &t : params_toks)
      {
        if (t.kind == tok_kind::identifier && t.text == ev.name)
        {
          found = true;
          break;
        }
      }
    }
    if (found)
    {
      captures.push_back(ev);
      captured_names.insert(ev.name);
    }
  }

  return captures;
}

// -----------------------------------------------------------------------
//  Lifted name generation
// -----------------------------------------------------------------------

static std::string
lifted_name(const std::string &enclosing, const std::string &nested)
{
  return "__esbmc_nested_" + enclosing + "__" + nested;
}

static std::string capture_global_name(
  const std::string &enclosing,
  const std::string &nested,
  const std::string &var)
{
  return "__esbmc_cap_" + enclosing + "__" + nested + "__" + var;
}

// -----------------------------------------------------------------------
//  Identifier rewriting
// -----------------------------------------------------------------------

// Replace identifiers in `text` according to `replacements` map.
static std::string rewrite_identifiers(
  const std::string &text,
  const std::map<std::string, std::string> &replacements)
{
  if (replacements.empty())
    return text;

  auto toks = tokenize(text);
  std::string result;
  for (const auto &t : toks)
  {
    if (t.kind == tok_kind::eof)
      break;

    if (t.kind == tok_kind::identifier)
    {
      auto it = replacements.find(t.text);
      if (it != replacements.end())
      {
        result += it->second;
        continue;
      }
    }
    result += t.text;
  }
  return result;
}

// Build a capture-replacement map for use in rewrite_identifiers.
static std::map<std::string, std::string> build_capture_replacements(
  const std::vector<local_var> &captures,
  bool fptr_mode,
  const std::string &enclosing,
  const std::string &func_name)
{
  std::map<std::string, std::string> m;
  for (const auto &c : captures)
  {
    if (fptr_mode)
      m[c.name] =
        "(*" + capture_global_name(enclosing, func_name, c.name) + ")";
    else
      m[c.name] = "(*__capture_" + c.name + ")";
  }
  return m;
}

static std::string rewrite_body(
  const std::string &body_text,
  const std::vector<local_var> &captures,
  bool fptr_mode,
  const std::string &enclosing,
  const std::string &func_name,
  const std::map<std::string, std::string> &sibling_renames =
    std::map<std::string, std::string>())
{
  auto replacements =
    build_capture_replacements(captures, fptr_mode, enclosing, func_name);
  // Merge sibling renames (original name -> lifted name)
  for (const auto &[k, v] : sibling_renames)
    replacements.insert({k, v});
  return rewrite_identifiers(body_text, replacements);
}

// Extract the inner parameter list (without outer parens) from params_text
static std::string inner_params(const std::string &params_text)
{
  // params_text is "(int a, int b)" -> "int a, int b"
  if (
    params_text.size() >= 2 && params_text.front() == '(' &&
    params_text.back() == ')')
    return params_text.substr(1, params_text.size() - 2);
  return params_text;
}

// -----------------------------------------------------------------------
//  Main transformation
// -----------------------------------------------------------------------

// Find all top-level function bodies and their nested functions.
// Returns nested_func entries sorted by def_start descending (so we can
// remove them from the source back-to-front without invalidating offsets).
static std::vector<nested_func>
find_nested_functions(const std::string &src, const std::vector<token> &toks)
{
  std::vector<nested_func> result;

  int brace_depth = 0;
  std::string current_func_name;
  size_t current_func_body_start_tok = 0;
  bool in_func_body = false;

  for (size_t i = 0; i < toks.size(); ++i)
  {
    const token &t = toks[i];

    if (t.kind == tok_kind::punctuation && t.text == "{")
    {
      ++brace_depth;
      continue;
    }

    if (t.kind == tok_kind::punctuation && t.text == "}")
    {
      --brace_depth;
      if (brace_depth == 0)
        in_func_body = false;
      continue;
    }

    // At file scope, detect function definitions
    if (brace_depth == 0 && t.kind == tok_kind::identifier)
    {
      nested_func dummy;
      size_t end_tok;
      if (try_parse_func_def(toks, i, "", dummy, end_tok, src))
      {
        current_func_name = dummy.name;
        // Find the opening brace token
        for (size_t j = i; j < end_tok; ++j)
        {
          if (toks[j].kind == tok_kind::punctuation && toks[j].text == "{")
          {
            current_func_body_start_tok = j;
            break;
          }
        }
        in_func_body = true;
        brace_depth = 1;
        i = current_func_body_start_tok;
        continue;
      }
    }

    // Inside a function body, look for nested function definitions
    if (in_func_body && brace_depth >= 1 && t.kind == tok_kind::identifier)
    {
      nested_func nf;
      size_t end_tok;
      if (try_parse_func_def(toks, i, current_func_name, nf, end_tok, src))
      {
        result.push_back(std::move(nf));
        // Skip past the nested function body
        // Update brace_depth: the body contains balanced braces
        // We need to count braces we skipped
        for (size_t j = i; j < end_tok; ++j)
        {
          if (toks[j].kind == tok_kind::punctuation)
          {
            if (toks[j].text == "{")
              ++brace_depth;
            else if (toks[j].text == "}")
              --brace_depth;
          }
        }
        i = end_tok - 1; // -1 because loop will ++i
        continue;
      }
    }
  }

  return result;
}

// Perform one pass of nested function transformation on `src`.
// Returns the transformed source, or empty string if no nested functions found.
static std::string transform_one_pass(const std::string &src)
{
  auto toks = tokenize(src);
  auto nested = find_nested_functions(src, toks);

  if (nested.empty())
    return {};

  // For each nested function, determine captures and fptr usage
  for (auto &nf : nested)
  {
    std::vector<local_var> enclosing_vars;
    std::set<std::string> nested_names;
    for (const auto &n : nested)
    {
      if (n.enclosing == nf.enclosing)
        nested_names.insert(n.name);
    }

    for (size_t i = 0; i < toks.size(); ++i)
    {
      if (toks[i].kind != tok_kind::identifier)
        continue;

      nested_func encl;
      size_t end_tok;
      if (try_parse_func_def(toks, i, "", encl, end_tok, src))
      {
        if (encl.name == nf.enclosing)
        {
          auto params = collect_params(encl.params_text);
          enclosing_vars.insert(
            enclosing_vars.end(), params.begin(), params.end());

          size_t body_start_tok = 0, body_end_tok = end_tok;
          for (size_t j = i; j < end_tok; ++j)
          {
            if (toks[j].kind == tok_kind::punctuation && toks[j].text == "{")
            {
              body_start_tok = j;
              break;
            }
          }

          auto locals = collect_local_vars(
            toks, body_start_tok, body_end_tok, nested_names);
          enclosing_vars.insert(
            enclosing_vars.end(), locals.begin(), locals.end());

          size_t def_start_tok = 0, def_end_tok_local = 0;
          for (size_t j = body_start_tok; j < body_end_tok; ++j)
          {
            if (toks[j].pos == nf.def_start)
            {
              def_start_tok = j;
              break;
            }
          }
          for (size_t j = def_start_tok; j < body_end_tok; ++j)
          {
            if (toks[j].pos >= nf.def_end)
            {
              def_end_tok_local = j;
              break;
            }
          }

          nf.used_as_fptr = check_fptr_use(
            toks,
            body_start_tok,
            body_end_tok,
            def_start_tok,
            def_end_tok_local,
            nf.name);
          break;
        }

        i = end_tok - 1;
      }
    }

    nf.captures = find_captures(nf.body_text, enclosing_vars, nf.params_text);
  }

  // Sort by def_start descending so we can modify the source back-to-front
  std::sort(
    nested.begin(),
    nested.end(),
    [](const nested_func &a, const nested_func &b) {
      return a.def_start > b.def_start;
    });

  // Step 1: Remove nested function definitions from the source
  std::string modified = src;

  for (const auto &nf : nested)
    modified.replace(nf.def_start, nf.def_end - nf.def_start, "");

  // Re-tokenize the modified source to find and transform call sites
  auto mod_toks = tokenize(modified);

  struct replacement
  {
    size_t start;
    size_t end;
    std::string text;
  };
  std::vector<replacement> replacements;

  for (const auto &nf : nested)
  {
    std::string lname = lifted_name(nf.enclosing, nf.name);

    // Find the enclosing function's body range in mod_toks to scope replacements
    size_t enc_body_start = 0, enc_body_end = mod_toks.size();
    for (size_t ei = 0; ei < mod_toks.size(); ++ei)
    {
      if (mod_toks[ei].kind != tok_kind::identifier)
        continue;
      nested_func enc_dummy;
      size_t enc_end;
      if (try_parse_func_def(mod_toks, ei, "", enc_dummy, enc_end, modified))
      {
        if (enc_dummy.name == nf.enclosing)
        {
          enc_body_start = ei;
          enc_body_end = enc_end;
          break;
        }
        ei = enc_end - 1;
      }
    }

    for (size_t i = enc_body_start; i < enc_body_end; ++i)
    {
      if (
        mod_toks[i].kind == tok_kind::identifier && mod_toks[i].text == nf.name)
      {
        size_t next = skip_ws(mod_toks, i + 1);

        if (nf.used_as_fptr)
        {
          replacements.push_back(
            {mod_toks[i].pos,
             mod_toks[i].pos + mod_toks[i].text.size(),
             lname});
        }
        else
        {
          if (
            next < mod_toks.size() &&
            mod_toks[next].kind == tok_kind::punctuation &&
            mod_toks[next].text == "(")
          {
            int pd = 1;
            size_t close = next + 1;
            while (close < mod_toks.size() && pd > 0)
            {
              if (mod_toks[close].kind == tok_kind::punctuation)
              {
                if (mod_toks[close].text == "(")
                  ++pd;
                else if (mod_toks[close].text == ")")
                  --pd;
              }
              ++close;
            }
            // Insert capture args BEFORE original args (after opening paren)
            std::string extra_args;
            for (const auto &cap : nf.captures)
            {
              if (!extra_args.empty())
                extra_args += ", ";
              extra_args += "&" + cap.name;
            }

            bool has_args = false;
            for (size_t j = next + 1; j < close - 1; ++j)
            {
              if (
                mod_toks[j].kind != tok_kind::whitespace &&
                mod_toks[j].kind != tok_kind::eof)
              {
                has_args = true;
                break;
              }
            }

            if (!extra_args.empty())
            {
              std::string insert_text =
                has_args ? extra_args + ", " : extra_args;
              // Insert right after the opening (
              size_t open_paren_end = mod_toks[next].pos + 1;
              replacements.push_back(
                {open_paren_end, open_paren_end, insert_text});
            }

            replacements.push_back(
              {mod_toks[i].pos,
               mod_toks[i].pos + mod_toks[i].text.size(),
               lname});
          }
        }
      }
    }
  }

  std::sort(
    replacements.begin(),
    replacements.end(),
    [](const replacement &a, const replacement &b) {
      return a.start > b.start;
    });

  for (const auto &r : replacements)
    modified.replace(r.start, r.end - r.start, r.text);

  // Step 2: Generate lifted functions, grouped by enclosing function
  std::map<std::string, std::string> per_enclosing_preamble;

  std::vector<nested_func> ordered = nested;
  std::sort(
    ordered.begin(),
    ordered.end(),
    [](const nested_func &a, const nested_func &b) {
      return a.def_start < b.def_start;
    });

  for (const auto &nf : ordered)
  {
    std::string lname = lifted_name(nf.enclosing, nf.name);
    std::string rewritten_body;
    std::string &preamble = per_enclosing_preamble[nf.enclosing];

    // Build sibling rename map: other nested funcs in the same enclosing
    std::map<std::string, std::string> sibling_renames;
    for (const auto &s : ordered)
    {
      if (s.enclosing == nf.enclosing && s.name != nf.name)
        sibling_renames[s.name] = lifted_name(s.enclosing, s.name);
    }

    if (nf.used_as_fptr)
    {
      for (const auto &cap : nf.captures)
      {
        std::string gname =
          capture_global_name(nf.enclosing, nf.name, cap.name);
        preamble += "static " + cap.type_text + " *" + gname + ";\n";
      }

      rewritten_body = rewrite_body(
        nf.body_text,
        nf.captures,
        true,
        nf.enclosing,
        nf.name,
        sibling_renames);
      preamble += "static " + nf.return_type + " " + lname + nf.params_text +
                  " " + rewritten_body + "\n";

      std::string setup;
      for (const auto &cap : nf.captures)
      {
        std::string gname =
          capture_global_name(nf.enclosing, nf.name, cap.name);
        setup += gname + " = &" + cap.name + "; ";
      }

      if (!setup.empty())
      {
        size_t pos = modified.find(lname);
        if (pos != std::string::npos)
        {
          size_t stmt_start = pos;
          while (stmt_start > 0 && modified[stmt_start - 1] != ';' &&
                 modified[stmt_start - 1] != '{' &&
                 modified[stmt_start - 1] != '}')
            --stmt_start;
          while (stmt_start < pos &&
                 std::isspace(static_cast<unsigned char>(modified[stmt_start])))
            ++stmt_start;

          modified.insert(stmt_start, setup);
        }
      }
    }
    else
    {
      rewritten_body = rewrite_body(
        nf.body_text,
        nf.captures,
        false,
        nf.enclosing,
        nf.name,
        sibling_renames);

      std::string params_inner = inner_params(nf.params_text);
      std::string trimmed_params;
      {
        size_t s = 0, e = params_inner.size();
        while (s < e &&
               std::isspace(static_cast<unsigned char>(params_inner[s])))
          ++s;
        while (e > s &&
               std::isspace(static_cast<unsigned char>(params_inner[e - 1])))
          --e;
        trimmed_params = params_inner.substr(s, e - s);
      }

      auto cap_repls =
        build_capture_replacements(nf.captures, false, nf.enclosing, nf.name);
      if (!cap_repls.empty())
        trimmed_params = rewrite_identifiers(trimmed_params, cap_repls);

      bool no_params = trimmed_params.empty() || trimmed_params == "void";
      std::string new_params;

      if (no_params && nf.captures.empty())
      {
        new_params = "(void)";
      }
      else
      {
        new_params = "(";
        for (size_t ci = 0; ci < nf.captures.size(); ++ci)
        {
          if (ci > 0)
            new_params += ", ";
          new_params +=
            nf.captures[ci].type_text + " *__capture_" + nf.captures[ci].name;
        }

        if (!no_params)
        {
          if (!nf.captures.empty())
            new_params += ", ";
          new_params += trimmed_params;
        }
        new_params += ")";
      }

      preamble += "static " + nf.return_type + " " + lname + new_params + " " +
                  rewritten_body + "\n";
    }
  }

  // Step 3: Insert each preamble immediately before its enclosing function.
  // Find enclosing function positions in `modified`, insert back-to-front.
  auto mod2_toks = tokenize(modified);

  struct preamble_insert
  {
    size_t pos;
    std::string text;
  };
  std::vector<preamble_insert> inserts;

  for (const auto &[enc_name, pre_text] : per_enclosing_preamble)
  {
    // Find the enclosing function definition in modified
    for (size_t i = 0; i < mod2_toks.size(); ++i)
    {
      if (mod2_toks[i].kind != tok_kind::identifier)
        continue;

      nested_func dummy;
      size_t end_tok;
      if (try_parse_func_def(mod2_toks, i, "", dummy, end_tok, modified))
      {
        if (dummy.name == enc_name)
        {
          inserts.push_back({mod2_toks[i].pos, pre_text + "\n"});
          break;
        }
        i = end_tok - 1;
      }
    }
  }

  // Sort by position descending and insert
  std::sort(
    inserts.begin(),
    inserts.end(),
    [](const preamble_insert &a, const preamble_insert &b) {
      return a.pos > b.pos;
    });

  for (const auto &ins : inserts)
    modified.insert(ins.pos, ins.text);

  return modified;
}

std::optional<file_operations::tmp_file>
transform_nested_functions(const std::string &source_path)
{
  std::ifstream in(source_path);
  if (!in)
    return std::nullopt;

  std::string src(
    (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();

  // Apply transformation iteratively to handle multi-level nesting.
  // Each pass lifts the outermost level of nested functions.
  // Limit iterations to prevent infinite loops on pathological input.
  bool transformed = false;
  for (int pass = 0; pass < 10; ++pass)
  {
    std::string result = transform_one_pass(src);
    if (result.empty())
      break;
    src = std::move(result);
    transformed = true;
  }

  if (!transformed)
    return std::nullopt;

  // Add #line directive at the top
  std::string output = "#line 1 \"" + source_path + "\"\n" + src;

  auto tmp = file_operations::create_tmp_file("esbmc-nested.%%%%-%%%%.c");
  if (!tmp.file())
    return std::nullopt;

  std::fputs(output.c_str(), tmp.file());
  std::fflush(tmp.file());

  return tmp;
}
