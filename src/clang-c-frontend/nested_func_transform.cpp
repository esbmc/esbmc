#include <clang-c-frontend/nested_func_transform.h>

#include <util/compiler_defs.h>
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/Basic/Version.inc>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Token.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
CC_DIAGNOSTIC_POP()

#include <util/message.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
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

// Range of a preprocessor directive in the raw source buffer.
struct pp_range
{
  size_t start; // byte offset of the leading '#'
  size_t end;   // byte offset one past the terminating '\n' (or src.size())
};

// Pre-scan the raw source for `#`-at-start-of-line directives.  Clang's raw
// lexer emits directives as individual tokens, but Layer B expects each
// directive to be one opaque `pp_directive` token that `skip_ws()` can skip
// wholesale.  We compute directive ranges with the same byte-level rules as
// the original hand-rolled tokenizer so offsets stay bit-exact.
static std::vector<pp_range> scan_pp_directives(const std::string &src)
{
  std::vector<pp_range> ranges;
  const size_t n = src.size();
  size_t i = 0;
  bool at_line_start = true;

  while (i < n)
  {
    const char c = src[i];
    if (c == '\n')
    {
      at_line_start = true;
      ++i;
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v')
    {
      ++i;
      continue;
    }
    if (c == '#' && at_line_start)
    {
      size_t start = i;
      ++i;
      while (i < n)
      {
        if (src[i] == '\n')
        {
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
      ranges.push_back({start, i});
      at_line_start = true;
      continue;
    }
    at_line_start = false;
    ++i;
  }
  return ranges;
}

// Map Clang's raw token kind to our coarse classification.
static tok_kind map_clang_kind(const clang::Token &t)
{
  using K = clang::tok::TokenKind;
  switch (t.getKind())
  {
  case K::raw_identifier:
    return tok_kind::identifier;
  case K::numeric_constant:
    return tok_kind::number;
  case K::string_literal:
  case K::wide_string_literal:
  case K::utf8_string_literal:
  case K::utf16_string_literal:
  case K::utf32_string_literal:
    return tok_kind::string_lit;
  case K::char_constant:
  case K::wide_char_constant:
  case K::utf8_char_constant:
  case K::utf16_char_constant:
  case K::utf32_char_constant:
    return tok_kind::char_lit;
  default:
    return tok_kind::punctuation;
  }
}

static std::string spelling_of(
  const clang::Token &t,
  const clang::SourceManager &sm,
  const clang::LangOptions &lo)
{
  if (t.is(clang::tok::raw_identifier))
    return t.getRawIdentifier().str();
  return clang::Lexer::getSpelling(t, sm, lo);
}

// Tokenize `src` into a flat vector using Clang's raw lexer.
// Comments are skipped.  String/char literals are kept as opaque blobs.
// Preprocessor directives are synthesized as single `pp_directive` tokens
// (byte-exact with the prior hand-rolled tokenizer) so Layer B consumers
// can keep skipping them wholesale via `skip_ws()`.
static std::vector<token> tokenize(const std::string &src)
{
  std::vector<token> tokens;
  const size_t n = src.size();

  // Permissive C/GNU mode: we only need structurally correct tokenization.
  // The user's real -std=... is honored later by Clang's own parse pass.
  clang::LangOptions LO;
  LO.C17 = 1;
  LO.GNUMode = 1;
  LO.GNUKeywords = 1;
  LO.Digraphs = 1;

#if CLANG_VERSION_MAJOR >= 21
  auto DiagOpts = std::make_shared<clang::DiagnosticOptions>();
#else
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
    new clang::DiagnosticOptions());
#endif
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(
    new clang::DiagnosticIDs());
  clang::DiagnosticsEngine diags(
    diag_ids,
#if CLANG_VERSION_MAJOR >= 21
    *DiagOpts,
#else
    &*DiagOpts,
#endif
    new clang::IgnoringDiagConsumer(),
    /*ShouldOwnClient=*/true);

  clang::FileSystemOptions fs_opts;
  clang::FileManager fm(fs_opts);
  clang::SourceManager sm(diags, fm);

  std::unique_ptr<llvm::MemoryBuffer> buf = llvm::MemoryBuffer::getMemBuffer(
    llvm::StringRef(src), /*BufferName=*/"", /*RequiresNullTerminator=*/true);
  llvm::MemoryBufferRef buf_ref = buf->getMemBufferRef();
  clang::FileID fid = sm.createFileID(std::move(buf));
  sm.setMainFileID(fid);

  clang::Lexer lex(fid, buf_ref, sm, LO);
  lex.SetCommentRetentionState(false);

  std::vector<pp_range> pp = scan_pp_directives(src);
  size_t pi = 0;
  auto emit_pending_pp = [&](size_t upto) {
    while (pi < pp.size() && pp[pi].end <= upto)
    {
      tokens.push_back(
        {tok_kind::pp_directive,
         src.substr(pp[pi].start, pp[pi].end - pp[pi].start),
         pp[pi].start});
      ++pi;
    }
  };

  clang::Token t;
  while (true)
  {
    lex.LexFromRawLexer(t);
    if (t.is(clang::tok::eof))
    {
      emit_pending_pp(n);
      break;
    }
    const size_t off = sm.getFileOffset(t.getLocation());
    // Swallow any raw token whose start offset is inside a pp-directive range.
    if (pi < pp.size() && off >= pp[pi].start && off < pp[pi].end)
      continue;
    emit_pending_pp(off);
    tokens.push_back({map_clang_kind(t), spelling_of(t, sm, LO), off});
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

// Try to parse a declaration at `toks[idx]`.  Best-effort heuristic covering:
//   type-keyword(s) [*]* name [= ...] [, name2 ...] ;
//   typedef-identifier name [= ...] ;            (IDENT IDENT followed by = ; [ or ,)
//   __label__ name, name, ... ;                  (consumed silently with a warning)
//   nested-function definition: ret-type name ( ... ) { ... }   (consumed, no vars)
//
// Returns a token index immediately past the consumed declaration.  If no
// declaration is recognized at `idx`, returns `idx` unchanged so callers can
// advance by one.  `out_vars` is cleared on entry and populated with any
// variables declared (possibly empty for __label__ or nested-function defs).
static size_t parse_decl_at(
  const std::vector<token> &toks,
  size_t idx,
  size_t end,
  const std::set<std::string> &nested_func_names,
  std::vector<local_var> &out_vars)
{
  out_vars.clear();

  if (idx >= end || toks[idx].kind != tok_kind::identifier)
    return idx;

  size_t try_idx = idx;
  std::string type_str;
  bool found_type = false;

  // Collect type tokens (type keywords and '*').
  while (try_idx < end)
  {
    size_t next = skip_ws(toks, try_idx);
    if (next >= end)
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
        if (tag < end && toks[tag].kind == tok_kind::identifier)
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
    size_t t1 = skip_ws(toks, idx);

    // __label__ declarations (GCC local labels): consume through ';'.
    if (
      t1 < end && toks[t1].kind == tok_kind::identifier &&
      toks[t1].text == "__label__")
    {
      log_warning(
        "GCC __label__ declarations are not supported with "
        "--gcc-nested-functions; results may be unsound");
      size_t scan = t1 + 1;
      while (scan < end)
      {
        if (toks[scan].kind == tok_kind::punctuation && toks[scan].text == ";")
          return scan + 1;
        ++scan;
      }
      return scan;
    }

    // Typedef heuristic: IDENT IDENT followed by = ; [ or ,
    if (
      t1 < end && toks[t1].kind == tok_kind::identifier &&
      !is_type_keyword(toks[t1].text) && !is_non_func_keyword(toks[t1].text))
    {
      size_t t2 = skip_ws(toks, t1 + 1);
      if (
        t2 < end && toks[t2].kind == tok_kind::identifier &&
        !is_type_keyword(toks[t2].text) &&
        !is_non_func_keyword(toks[t2].text) &&
        !nested_func_names.count(toks[t2].text))
      {
        size_t t3 = skip_ws(toks, t2 + 1);
        if (
          t3 < end && toks[t3].kind == tok_kind::punctuation &&
          (toks[t3].text == "=" || toks[t3].text == ";" ||
           toks[t3].text == "[" || toks[t3].text == ","))
        {
          out_vars.push_back({toks[t1].text, toks[t2].text});
          size_t scan = t3;
          while (scan < end)
          {
            if (
              toks[scan].kind == tok_kind::punctuation &&
              toks[scan].text == ";")
              return scan + 1;
            ++scan;
          }
          return scan;
        }
      }
    }
    return idx;
  }

  // found_type: parse one or more: name [= expr] [, ...]  ;  OR a nested
  // function definition (skipped, no vars added).
  while (true)
  {
    size_t name_idx = skip_ws(toks, try_idx);
    if (
      name_idx >= end || toks[name_idx].kind != tok_kind::identifier ||
      is_type_keyword(toks[name_idx].text))
      break;

    std::string var_name = toks[name_idx].text;

    // Nested function definition: skip params + body, do not add as var.
    if (nested_func_names.count(var_name))
    {
      size_t skip = skip_ws(toks, name_idx + 1);
      if (
        skip < end && toks[skip].kind == tok_kind::punctuation &&
        toks[skip].text == "(")
      {
        int pd = 1;
        ++skip;
        while (skip < end && pd > 0)
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
        size_t brace = skip_ws(toks, skip);
        if (
          brace < end && toks[brace].kind == tok_kind::punctuation &&
          toks[brace].text == "{")
        {
          int bd = 1;
          ++brace;
          while (brace < end && bd > 0)
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
          goto done;
        }
      }
      break;
    }

    // Check what follows: = , ; or [ signals a declaration.
    size_t after = skip_ws(toks, name_idx + 1);
    if (after >= end)
      break;

    bool is_var = toks[after].kind == tok_kind::punctuation &&
                  (toks[after].text == "=" || toks[after].text == "," ||
                   toks[after].text == ";" || toks[after].text == "[");
    if (!is_var)
      break;

    out_vars.push_back({type_str, var_name});

    // Skip to the next ',' or ';' at depth 0.
    size_t scan = after;
    int skip_depth = 0;
    bool more = false;
    while (scan < end)
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
          more = true;
          break;
        }
        else if (skip_depth == 0 && toks[scan].text == ";")
        {
          try_idx = scan + 1;
          goto done;
        }
      }
      ++scan;
    }
    if (!more)
      break;
  }

done:
  // Consume at least the type tokens even if no name followed; this matches
  // the prior walker behaviour that treated `int` anywhere as a decl marker.
  return try_idx;
}

// Collect local variable declarations from a function body at depth 1 only.
// Used to enumerate enclosing-function locals before the scope-aware walker
// analyses nested-function bodies.
static std::vector<local_var> collect_local_vars(
  const std::vector<token> &toks,
  size_t body_start_tok,
  size_t body_end_tok,
  const std::set<std::string> &nested_func_names)
{
  std::vector<local_var> vars;
  std::vector<local_var> decl;
  int depth = 0;
  size_t idx = body_start_tok;

  while (idx < body_end_tok)
  {
    if (toks[idx].kind == tok_kind::punctuation)
    {
      if (toks[idx].text == "{")
      {
        ++depth;
        ++idx;
        continue;
      }
      if (toks[idx].text == "}")
      {
        --depth;
        ++idx;
        continue;
      }
    }

    if (depth == 1)
    {
      size_t next =
        parse_decl_at(toks, idx, body_end_tok, nested_func_names, decl);
      if (next != idx)
      {
        for (auto &v : decl)
          vars.push_back(std::move(v));
        idx = next;
        continue;
      }
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

    // Only treat identifiers outside brackets as potential param names.
    // Type keywords go into type_parts, not last_ident.
    if (t.kind == tok_kind::identifier && depth == 1 && bracket_depth == 0)
    {
      if (is_type_keyword(t.text))
      {
        // Type keyword is part of the type, not a param name
        if (!last_ident.empty())
          type_parts.push_back(last_ident);
        last_ident.clear();
        type_parts.push_back(t.text);
      }
      else
      {
        if (!last_ident.empty())
          type_parts.push_back(last_ident);
        last_ident = t.text;
      }
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

// -----------------------------------------------------------------------
//  Scope-aware walker
// -----------------------------------------------------------------------

// Find the preceding non-trivia token (skipping pp_directive tokens).
// Returns `begin` if no such token exists at or after `begin`.
static size_t
skip_ws_back(const std::vector<token> &toks, size_t idx, size_t begin)
{
  if (idx <= begin)
    return begin;
  size_t j = idx - 1;
  while (j > begin && toks[j].kind == tok_kind::pp_directive)
    --j;
  return j;
}

// Given `idx` points at a control-flow keyword followed (after skip_ws) by
// "(", return the token index immediately past the end of the statement it
// governs.  Recurses on nested control flow in single-statement bodies.
// Returns `end` if malformed input.
static size_t
find_control_stmt_end(const std::vector<token> &toks, size_t idx, size_t end)
{
  if (idx >= end)
    return end;

  size_t i = idx + 1;
  i = skip_ws(toks, i);
  if (i >= end || toks[i].kind != tok_kind::punctuation || toks[i].text != "(")
    return end;

  // Scan to the matching ')'.
  int pd = 1;
  ++i;
  while (i < end && pd > 0)
  {
    if (toks[i].kind == tok_kind::punctuation)
    {
      if (toks[i].text == "(")
        ++pd;
      else if (toks[i].text == ")")
        --pd;
    }
    ++i;
  }

  // Body start.
  i = skip_ws(toks, i);
  if (i >= end)
    return end;

  // Block body: scan to matching '}'.
  if (toks[i].kind == tok_kind::punctuation && toks[i].text == "{")
  {
    int bd = 1;
    ++i;
    while (i < end && bd > 0)
    {
      if (toks[i].kind == tok_kind::punctuation)
      {
        if (toks[i].text == "{")
          ++bd;
        else if (toks[i].text == "}")
          --bd;
      }
      ++i;
    }
    return i;
  }

  // Nested control flow as body: recurse.
  if (toks[i].kind == tok_kind::identifier)
  {
    const std::string &kw = toks[i].text;
    if (kw == "for" || kw == "while" || kw == "if" || kw == "switch")
      return find_control_stmt_end(toks, i, end);
  }

  // Single-statement body: scan to ';' at paren/brace depth 0.
  int d = 0;
  while (i < end)
  {
    if (toks[i].kind == tok_kind::punctuation)
    {
      if (toks[i].text == "(" || toks[i].text == "[" || toks[i].text == "{")
        ++d;
      else if (
        toks[i].text == ")" || toks[i].text == "]" || toks[i].text == "}")
        --d;
      else if (d == 0 && toks[i].text == ";")
        return i + 1;
    }
    ++i;
  }
  return i;
}

// Scope-aware walker over a token range.  Supports block scopes (`{ ... }`
// in statement position) and C99 `for`-init scopes (the init declaration is
// live through the loop body).  Non-scope `{` braces (compound literals,
// designated initializers, GCC statement-expressions, struct/union/enum
// member lists) are tracked but do not push or add to any scope.
struct scope_walker
{
  const std::vector<token> &toks;
  size_t begin;
  size_t end;
  const std::set<std::string> &nested_names;

  // Scopes stack; innermost scope last.  Always at least one entry (root).
  std::vector<std::set<std::string>> scopes;
  // Aligned with every `{` seen; true if it pushed a scope onto `scopes`.
  std::vector<bool> brace_pushed_scope;
  // Depth of currently-open non-scope braces (used to suppress add_decl
  // inside struct/union/enum member lists, compound literals, and GCC
  // statement-expressions).
  int nonscope_depth = 0;
  // Indices at which a `for`-scope should be popped (before processing the
  // token at that index).
  std::vector<size_t> for_pop_at;

  scope_walker(
    const std::vector<token> &t,
    size_t b,
    size_t e,
    const std::set<std::string> &nn,
    std::set<std::string> seed = {})
    : toks(t), begin(b), end(e), nested_names(nn)
  {
    scopes.push_back(std::move(seed));
  }

  // Any scope on the stack declares `name`?
  bool declares(const std::string &name) const
  {
    for (const auto &s : scopes)
      if (s.count(name))
        return true;
    return false;
  }

  // Add `name` to the innermost scope.  No-op inside a non-scope brace.
  void add_decl(const std::string &name)
  {
    if (nonscope_depth > 0)
      return;
    scopes.back().insert(name);
  }

  // Call before inspecting `toks[idx]`.  Pops any for-scopes whose governed
  // statement ended at or before this index.
  void pre(size_t idx)
  {
    while (!for_pop_at.empty() && for_pop_at.back() <= idx)
    {
      for_pop_at.pop_back();
      if (scopes.size() > 1)
        scopes.pop_back();
    }
  }

  // Does `toks[idx]` (which must be `{`) stand in statement position?
  // Preceding non-trivia token decides: `=` / `,` / `(` / `struct` / `union`
  // / `enum` (or an identifier tag following one of those) mean non-scope.
  bool is_block_brace(size_t idx) const
  {
    if (idx <= begin)
      return true;
    size_t j = skip_ws_back(toks, idx, begin);
    if (j == idx)
      return true;
    const token &p = toks[j];
    if (p.kind == tok_kind::punctuation)
    {
      if (p.text == "=" || p.text == "," || p.text == "(")
        return false;
      return true;
    }
    if (p.kind == tok_kind::identifier)
    {
      if (p.text == "struct" || p.text == "union" || p.text == "enum")
        return false;
      // tag-identifier after struct/union/enum: also a member list.
      if (j > begin)
      {
        size_t k = skip_ws_back(toks, j, begin);
        if (
          k != j && toks[k].kind == tok_kind::identifier &&
          (toks[k].text == "struct" || toks[k].text == "union" ||
           toks[k].text == "enum"))
          return false;
      }
      return true;
    }
    return true;
  }

  // Callers invoke these as they step over tokens.
  void on_open_brace(size_t idx)
  {
    bool scope = is_block_brace(idx);
    brace_pushed_scope.push_back(scope);
    if (scope)
      scopes.push_back({});
    else
      ++nonscope_depth;
  }
  void on_close_brace()
  {
    if (brace_pushed_scope.empty())
      return;
    bool was_scope = brace_pushed_scope.back();
    brace_pushed_scope.pop_back();
    if (was_scope)
    {
      if (scopes.size() > 1)
        scopes.pop_back();
    }
    else if (nonscope_depth > 0)
    {
      --nonscope_depth;
    }
  }
  void enter_for_scope(size_t stmt_end)
  {
    scopes.push_back({});
    for_pop_at.push_back(stmt_end);
  }
};

// Walk a body token range with a scope-aware walker, invoking `on_use` for
// every identifier use at a point where no scope currently declares it, and
// recording declarations in the walker.  Declarations are processed at the
// statement head by pre-adding their declared names to the current scope
// before processing any individual token — so initializer expressions
// inside `int x = expr;` are still visited as uses.
template <class OnUse>
static void scope_walk(scope_walker &w, OnUse on_use)
{
  std::vector<local_var> decl_buf;
  size_t decl_end = w.begin;
  size_t idx = w.begin;
  while (idx < w.end)
  {
    w.pre(idx);
    const token &t = w.toks[idx];

    if (t.kind == tok_kind::punctuation)
    {
      if (t.text == "{")
      {
        w.on_open_brace(idx);
        ++idx;
        continue;
      }
      if (t.text == "}")
      {
        w.on_close_brace();
        ++idx;
        continue;
      }
    }

    // `for` / `while` / `if` / `switch` — push a scope that extends through
    // the governed statement.  Only `for` permits a declaration in the
    // init, but the other keywords in C99 cannot introduce new names, so a
    // pushed empty scope is harmless there and keeps scope-tracking uniform.
    if (t.kind == tok_kind::identifier)
    {
      const std::string &kw = t.text;
      if (kw == "for" || kw == "while" || kw == "if" || kw == "switch")
      {
        size_t after = skip_ws(w.toks, idx + 1);
        if (
          after < w.end && w.toks[after].kind == tok_kind::punctuation &&
          w.toks[after].text == "(")
        {
          size_t stmt_end = find_control_stmt_end(w.toks, idx, w.end);
          w.enter_for_scope(stmt_end);
          ++idx;
          continue;
        }
      }
    }

    // At the head of a potential declaration, pre-add the declared names
    // to the current scope so any reference to them (including the name
    // tokens themselves) is seen as "declared".  Do NOT skip the decl's
    // tokens — the initializer may contain identifier uses that must be
    // processed normally.
    if (idx >= decl_end)
    {
      size_t next = parse_decl_at(w.toks, idx, w.end, w.nested_names, decl_buf);
      if (next != idx)
      {
        for (const auto &v : decl_buf)
          w.add_decl(v.name);
        decl_end = next;
      }
    }

    // Identifier use — caller decides what to do with it.
    if (t.kind == tok_kind::identifier)
      on_use(idx, t.text);
    ++idx;
  }
}

// Identify which enclosing variables are captured by a nested function.
// Uses a scope-aware walker so that inner-block, function-parameter, or
// `for`-init declarations correctly shadow outer names.
static std::vector<local_var> find_captures(
  const std::string &body_text,
  const std::vector<local_var> &enclosing_vars,
  const std::string &params_text)
{
  auto body_toks = tokenize(body_text);
  auto params_toks = tokenize(params_text);
  auto param_vars = collect_params(params_text);

  std::set<std::string> param_names;
  for (const auto &p : param_vars)
    param_names.insert(p.name);

  // Index enclosing names for O(1) membership.
  std::set<std::string> enclosing_names;
  for (const auto &ev : enclosing_vars)
    enclosing_names.insert(ev.name);

  std::set<std::string> referenced;

  std::set<std::string> no_nested;
  scope_walker w(body_toks, 0, body_toks.size(), no_nested, param_names);
  scope_walk(w, [&](size_t idx, const std::string &name) {
    if (!enclosing_names.count(name))
      return;
    if (w.declares(name))
      return;
    // Don't treat an identifier after a member-access operator as a
    // reference to an outer variable.
    if (idx > 0)
    {
      size_t prev = skip_ws_back(body_toks, idx, 0);
      if (
        prev != idx && body_toks[prev].kind == tok_kind::punctuation &&
        (body_toks[prev].text == "." || body_toks[prev].text == "->"))
        return;
    }
    referenced.insert(name);
  });

  // Also check params_text for uses inside VLA parameter types.
  for (const auto &t : params_toks)
    if (
      t.kind == tok_kind::identifier && enclosing_names.count(t.text) &&
      !param_names.count(t.text))
      referenced.insert(t.text);

  // Emit in the original enclosing_vars order.
  std::vector<local_var> captures;
  for (const auto &ev : enclosing_vars)
    if (referenced.count(ev.name))
      captures.push_back(ev);
  return captures;
}

// -----------------------------------------------------------------------
//  Lifted name generation
// -----------------------------------------------------------------------

// Length-prefixed encoding ensures the mapping is injective: different
// (enclosing, nested) pairs always produce different lifted names, even
// when identifier names contain underscores.
static std::string
lifted_name(const std::string &enclosing, const std::string &nested)
{
  return "__esbmc_nested_" + std::to_string(enclosing.size()) + "_" +
         enclosing + "_" + std::to_string(nested.size()) + "_" + nested;
}

static std::string capture_global_name(
  const std::string &enclosing,
  const std::string &nested,
  const std::string &var)
{
  return "__esbmc_cap_" + std::to_string(enclosing.size()) + "_" + enclosing +
         "_" + std::to_string(nested.size()) + "_" + nested + "_" +
         std::to_string(var.size()) + "_" + var;
}

static std::string capture_param_name(
  const std::string &enclosing,
  const std::string &nested,
  const std::string &var)
{
  return "__esbmc_p_" + std::to_string(enclosing.size()) + "_" + enclosing +
         "_" + std::to_string(nested.size()) + "_" + nested + "_" +
         std::to_string(var.size()) + "_" + var;
}

// -----------------------------------------------------------------------
//  Identifier rewriting
// -----------------------------------------------------------------------

// Replace identifiers in `text` according to `replacements` map.
// Replace identifiers in `text` according to `replacements`.  Scope-aware:
// names listed in `scope_aware_names` are replaced only when no enclosing
// scope on the walker's stack declares them (so inner-block declarations
// and `for`-init declarations properly shadow outer captures).  Names not
// in `scope_aware_names` (e.g. sibling-nested-function renames) are
// replaced unconditionally — those are globally unique in the transform's
// name scheme and never introduced as local names.  The after-member-op
// heuristic (do not replace identifiers following `.` or `->`) always
// applies.
static std::string rewrite_identifiers(
  const std::string &text,
  const std::map<std::string, std::string> &replacements,
  const std::set<std::string> *scope_aware_names = nullptr,
  std::set<std::string> seed_scope = {})
{
  if (replacements.empty())
    return text;

  auto toks = tokenize(text);
  std::set<std::string> no_nested;
  scope_walker w(toks, 0, toks.size(), no_nested, std::move(seed_scope));

  std::vector<local_var> decl_buf;
  size_t decl_end = 0; // tokens up to (not including) this index are part of
                       // a decl whose names are already added to the scope.

  std::string result;
  size_t cursor = 0;
  bool after_member_op = false;

  for (size_t ti = 0; ti < toks.size();)
  {
    w.pre(ti);
    const auto &t = toks[ti];
    if (t.kind == tok_kind::eof)
      break;

    // Structural: `{`, `}`.
    if (t.kind == tok_kind::punctuation)
    {
      if (t.text == "{")
      {
        w.on_open_brace(ti);
        if (t.pos > cursor)
          result += text.substr(cursor, t.pos - cursor);
        result += t.text;
        cursor = t.pos + t.text.size();
        after_member_op = false;
        ++ti;
        continue;
      }
      if (t.text == "}")
      {
        w.on_close_brace();
        if (t.pos > cursor)
          result += text.substr(cursor, t.pos - cursor);
        result += t.text;
        cursor = t.pos + t.text.size();
        after_member_op = false;
        ++ti;
        continue;
      }
    }

    // `for`/`while`/`if`/`switch` — enter condition-governed scope.
    if (t.kind == tok_kind::identifier)
    {
      const std::string &kw = t.text;
      if (kw == "for" || kw == "while" || kw == "if" || kw == "switch")
      {
        size_t after = skip_ws(toks, ti + 1);
        if (
          after < toks.size() && toks[after].kind == tok_kind::punctuation &&
          toks[after].text == "(")
        {
          size_t stmt_end = find_control_stmt_end(toks, ti, toks.size());
          w.enter_for_scope(stmt_end);
        }
      }
    }

    // At the start of a potential declaration, run parse_decl_at once and
    // pre-add the declared names to the current scope so that the name
    // tokens, when visited below, are seen as "declared" and not replaced.
    if (ti >= decl_end)
    {
      size_t next = parse_decl_at(toks, ti, toks.size(), no_nested, decl_buf);
      if (next != ti)
      {
        for (const auto &v : decl_buf)
          w.add_decl(v.name);
        decl_end = next;
      }
    }

    // Emit interstitial bytes preceding this token.
    if (t.pos > cursor)
      result += text.substr(cursor, t.pos - cursor);

    bool replaced = false;
    if (t.kind == tok_kind::identifier && !after_member_op)
    {
      auto it = replacements.find(t.text);
      if (it != replacements.end())
      {
        bool shadowed = false;
        if (scope_aware_names && scope_aware_names->count(t.text))
          shadowed = w.declares(t.text);
        if (!shadowed)
        {
          result += it->second;
          replaced = true;
        }
      }
    }
    if (!replaced)
      result += t.text;

    if (t.kind == tok_kind::punctuation && (t.text == "." || t.text == "->"))
      after_member_op = true;
    else
      after_member_op = false;

    cursor = t.pos + t.text.size();
    ++ti;
  }

  if (cursor < text.size())
    result += text.substr(cursor);
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
      m[c.name] = "(*" + capture_param_name(enclosing, func_name, c.name) + ")";
  }
  return m;
}

// Rewrite capture references inside a nested function body.  Sibling
// renames and inter-sibling call arg injection are handled separately by
// rewrite_nested_calls after this pass.
static std::string rewrite_body(
  const std::string &body_text,
  const std::string &params_text,
  const std::vector<local_var> &captures,
  bool fptr_mode,
  const std::string &enclosing,
  const std::string &func_name)
{
  auto replacements =
    build_capture_replacements(captures, fptr_mode, enclosing, func_name);

  // Seed the walker's root scope with parameter names so a parameter that
  // happens to shadow an enclosing capture name is not rewritten.
  std::set<std::string> param_seed;
  for (const auto &p : collect_params(params_text))
    param_seed.insert(p.name);

  std::set<std::string> scope_aware_names;
  for (const auto &c : captures)
    scope_aware_names.insert(c.name);

  return rewrite_identifiers(
    body_text, replacements, &scope_aware_names, std::move(param_seed));
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

// Collect sibling nested-function names that `nf.body_text` directly calls,
// using the scope-aware walker so that a local variable named `b` that
// shadows a sibling nested function also named `b` does not produce a
// spurious call-graph edge.  Only identifier-name calls followed by `(`
// count; indirect calls via fptrs are silently skipped (out of scope).
static std::set<std::string> find_sibling_callees(
  const std::string &body_text,
  const std::string &params_text,
  const std::set<std::string> &sibling_names)
{
  std::set<std::string> result;
  if (sibling_names.empty())
    return result;

  auto body_toks = tokenize(body_text);
  auto param_vars = collect_params(params_text);

  std::set<std::string> param_names;
  for (const auto &p : param_vars)
    param_names.insert(p.name);

  // Do not pass sibling_names as "nested_names" to the scope walker's
  // decl parser: sibling definitions themselves are NOT present in the
  // body_text (the transform operates on each nested function's body in
  // isolation), so nested_names is empty for local-var detection here.
  std::set<std::string> no_nested;
  scope_walker w(body_toks, 0, body_toks.size(), no_nested, param_names);

  scope_walk(w, [&](size_t idx, const std::string &name) {
    if (!sibling_names.count(name))
      return;
    if (w.declares(name))
      return;
    size_t next = skip_ws(body_toks, idx + 1);
    if (
      next < body_toks.size() &&
      body_toks[next].kind == tok_kind::punctuation &&
      body_toks[next].text == "(")
      result.insert(name);
  });

  return result;
}

// Propagate captures along the sibling call graph to a fixed point.
// After this, each nested function's `captures` list holds its *effective*
// captures: direct captures plus the captures of every sibling it can
// transitively call.  Emission order matches the enclosing_vars order so
// lifted-signature emission, call-site arg injection, and capture-param
// names line up across all three rewriters.
static void compute_effective_captures(
  std::vector<nested_func> &nested,
  const std::map<std::string, std::vector<local_var>> &enc_vars_by_name)
{
  // Group nested indices by their enclosing function.
  std::map<std::string, std::vector<size_t>> by_enc;
  for (size_t i = 0; i < nested.size(); ++i)
    by_enc[nested[i].enclosing].push_back(i);

  for (const auto &[enc_name, indices] : by_enc)
  {
    auto it = enc_vars_by_name.find(enc_name);
    if (it == enc_vars_by_name.end())
      continue;
    const auto &enc_vars = it->second;

    // Map variable name -> its position in enc_vars for order preservation.
    std::map<std::string, size_t> pos_of;
    for (size_t p = 0; p < enc_vars.size(); ++p)
      pos_of[enc_vars[p].name] = p;

    std::set<std::string> sibling_names;
    for (size_t i : indices)
      sibling_names.insert(nested[i].name);

    // Direct call graph.
    std::map<std::string, std::set<std::string>> callees;
    for (size_t i : indices)
      callees[nested[i].name] = find_sibling_callees(
        nested[i].body_text, nested[i].params_text, sibling_names);

    // Effective capture set per function name.
    std::map<std::string, std::set<std::string>> eff;
    for (size_t i : indices)
      for (const auto &c : nested[i].captures)
        eff[nested[i].name].insert(c.name);

    // Fixed-point propagation.  Finite lattice + monotone growth => must
    // converge.
    bool changed = true;
    while (changed)
    {
      changed = false;
      for (size_t i : indices)
      {
        auto &my_eff = eff[nested[i].name];
        for (const auto &callee : callees[nested[i].name])
        {
          const auto &callee_eff = eff[callee];
          for (const auto &c : callee_eff)
            if (my_eff.insert(c).second)
              changed = true;
        }
      }
    }

    // Write back.  Preserve enc_vars positional order so downstream
    // emission order is deterministic across call sites and lifted
    // signatures.
    for (size_t i : indices)
    {
      std::vector<std::pair<size_t, const local_var *>> ordered;
      for (const auto &name : eff[nested[i].name])
      {
        auto pit = pos_of.find(name);
        if (pit != pos_of.end())
          ordered.push_back({pit->second, &enc_vars[pit->second]});
      }
      std::sort(
        ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });
      std::vector<local_var> new_caps;
      new_caps.reserve(ordered.size());
      for (const auto &[p, lv] : ordered)
        new_caps.push_back(*lv);
      nested[i].captures = std::move(new_caps);
    }
  }
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

// A single contiguous replacement in a string: replace [start, end) with `text`.
struct rewrite_op
{
  size_t start;
  size_t end;
  std::string text;
};

// Apply rewrites back-to-front (so earlier offsets stay valid during each
// replacement).
static std::string apply_rewrites(std::string s, std::vector<rewrite_op> ops)
{
  std::sort(
    ops.begin(), ops.end(), [](const rewrite_op &a, const rewrite_op &b) {
      return a.start > b.start;
    });
  for (const auto &op : ops)
    s.replace(op.start, op.end - op.start, op.text);
  return s;
}

// Plan rewrites for sibling-nested-function calls within a token range.
//   - Direct-call sibling: inject `source_for(cap.name)` for each capture
//     of the callee at the start of the argument list, and rename the
//     identifier to the callee's lifted name.
//   - Fptr-mode sibling: rename the identifier to the lifted name.  No
//     args are injected (fptr-mode callees read captures from statics).
// Scope-aware: skips call sites where the sibling name is shadowed by a
// local variable or parameter.  Offsets are into the body string of whose
// tokens are supplied.
static std::vector<rewrite_op> plan_nested_call_rewrites(
  const std::vector<token> &body_toks,
  size_t tok_begin,
  size_t tok_end,
  const std::vector<const nested_func *> &siblings,
  const std::function<std::string(const std::string &)> &source_for)
{
  std::vector<rewrite_op> out;
  if (siblings.empty())
    return out;

  std::map<std::string, const nested_func *> by_name;
  for (const nested_func *s : siblings)
    by_name[s->name] = s;

  std::set<std::string> no_nested;
  scope_walker w(body_toks, tok_begin, tok_end, no_nested);

  scope_walk(w, [&](size_t i, const std::string &name) {
    auto it = by_name.find(name);
    if (it == by_name.end())
      return;
    const nested_func *callee = it->second;

    // Don't rewrite an identifier following a member-access operator.
    size_t prev = skip_ws_back(body_toks, i, tok_begin);
    if (
      prev != i && body_toks[prev].kind == tok_kind::punctuation &&
      (body_toks[prev].text == "." || body_toks[prev].text == "->"))
      return;

    std::string lname = lifted_name(callee->enclosing, callee->name);

    if (callee->used_as_fptr)
    {
      out.push_back(
        {body_toks[i].pos, body_toks[i].pos + body_toks[i].text.size(), lname});
      return;
    }

    size_t next = skip_ws(body_toks, i + 1);
    if (
      next >= tok_end || body_toks[next].kind != tok_kind::punctuation ||
      body_toks[next].text != "(")
      return;

    // Match the closing ')'.
    int pd = 1;
    size_t close = next + 1;
    while (close < tok_end && pd > 0)
    {
      if (body_toks[close].kind == tok_kind::punctuation)
      {
        if (body_toks[close].text == "(")
          ++pd;
        else if (body_toks[close].text == ")")
          --pd;
      }
      ++close;
    }

    std::string extra;
    for (const auto &cap : callee->captures)
    {
      if (!extra.empty())
        extra += ", ";
      extra += source_for(cap.name);
    }

    bool has_args = false;
    for (size_t j = next + 1; j + 1 < close; ++j)
    {
      if (
        body_toks[j].kind != tok_kind::whitespace &&
        body_toks[j].kind != tok_kind::pp_directive &&
        body_toks[j].kind != tok_kind::eof)
      {
        has_args = true;
        break;
      }
    }

    if (!extra.empty())
    {
      std::string insert = has_args ? extra + ", " : extra;
      size_t open_paren_end = body_toks[next].pos + 1;
      out.push_back({open_paren_end, open_paren_end, insert});
    }

    out.push_back(
      {body_toks[i].pos, body_toks[i].pos + body_toks[i].text.size(), lname});
  });

  return out;
}

// Rewrite sibling-nested-function calls in a standalone body string
// (local offsets).  See plan_nested_call_rewrites for semantics.
static std::string rewrite_nested_calls(
  const std::string &body,
  const std::vector<const nested_func *> &siblings,
  const std::function<std::string(const std::string &)> &source_for)
{
  auto toks = tokenize(body);
  auto ops =
    plan_nested_call_rewrites(toks, 0, toks.size(), siblings, source_for);
  return apply_rewrites(body, std::move(ops));
}

// Perform one pass of nested function transformation on `src`.
// Returns the transformed source, or empty string if no nested functions found.
static std::string transform_one_pass(const std::string &src)
{
  auto toks = tokenize(src);
  auto nested = find_nested_functions(src, toks);

  if (nested.empty())
    return {};

  // Cache each enclosing function's (params + depth-1 locals) so the
  // transitive-capture propagator can reuse it without re-deriving.
  std::map<std::string, std::vector<local_var>> enc_vars_by_name;

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
    enc_vars_by_name[nf.enclosing] = std::move(enclosing_vars);
  }

  // Propagate captures through sibling calls: if A calls B and B captures
  // `x`, A must also carry `x` so it can forward at the call site.
  compute_effective_captures(nested, enc_vars_by_name);

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

  // Re-tokenize the modified source to find and transform call sites.
  auto mod_toks = tokenize(modified);

  // Rewrite calls within each enclosing function's body, passing `&x` as
  // the source expression for a capture `x` (enclosing-scope variables are
  // in local scope here).  Scope-aware to avoid rewriting names shadowed
  // by local variables in the enclosing body.
  std::vector<rewrite_op> call_ops;
  std::set<std::string> seen_enclosing;
  for (const auto &nf : nested)
  {
    if (!seen_enclosing.insert(nf.enclosing).second)
      continue;

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

    std::vector<const nested_func *> siblings_in_enc;
    for (const auto &s : nested)
      if (s.enclosing == nf.enclosing)
        siblings_in_enc.push_back(&s);

    auto ops = plan_nested_call_rewrites(
      mod_toks,
      enc_body_start,
      enc_body_end,
      siblings_in_enc,
      [](const std::string &name) { return "&" + name; });

    call_ops.insert(
      call_ops.end(),
      std::make_move_iterator(ops.begin()),
      std::make_move_iterator(ops.end()));
  }

  modified = apply_rewrites(std::move(modified), std::move(call_ops));

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

    // All nested-function siblings in this enclosing (including `nf`
    // itself — a recursive self-call is rewritten to its lifted name).
    std::vector<const nested_func *> siblings_in_enc;
    for (const auto &s : ordered)
      if (s.enclosing == nf.enclosing)
        siblings_in_enc.push_back(&s);

    if (nf.used_as_fptr)
    {
      for (const auto &cap : nf.captures)
      {
        std::string gname =
          capture_global_name(nf.enclosing, nf.name, cap.name);
        preamble += "static " + cap.type_text + " *" + gname + ";\n";
      }

      rewritten_body = rewrite_body(
        nf.body_text, nf.params_text, nf.captures, true, nf.enclosing, nf.name);

      // Rewrite sibling calls inside this lifted body.  An fptr-mode
      // caller reads its captures via statics, so to forward a capture
      // `x` we pass the caller's static pointer (already of type `T *`).
      rewritten_body = rewrite_nested_calls(
        rewritten_body, siblings_in_enc, [&nf](const std::string &name) {
          return capture_global_name(nf.enclosing, nf.name, name);
        });

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
        nf.params_text,
        nf.captures,
        false,
        nf.enclosing,
        nf.name);

      // Rewrite sibling calls inside this lifted body.  A direct-call
      // caller has `T *cap_x` as a parameter; forward that pointer
      // directly when calling siblings that also need `x`.
      rewritten_body = rewrite_nested_calls(
        rewritten_body, siblings_in_enc, [&nf](const std::string &name) {
          return capture_param_name(nf.enclosing, nf.name, name);
        });

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
            nf.captures[ci].type_text + " *" +
            capture_param_name(nf.enclosing, nf.name, nf.captures[ci].name);
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
