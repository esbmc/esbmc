// #7797: <regex> models everything around the match exactly; the match itself
// is nondeterministic, so a match_results only has to agree with what the call
// reported.
#include <regex>
#include <cassert>
#include <string>

int main()
{
  std::regex re("(a)(b)c");
  assert(re.mark_count() == 2);
  assert(re.flags() == std::regex_constants::ECMAScript);

  // A group opened by "(?" is not a marked sub-expression.
  std::regex non_marking("(?:ab)(c)");
  assert(non_marking.mark_count() == 1);

  // An escaped parenthesis is a literal, not a group.
  std::regex escaped("\\(a\\)");
  assert(escaped.mark_count() == 0);

  std::regex icase_re("abc", std::regex_constants::icase);
  assert((icase_re.flags() & std::regex_constants::icase) != 0);

  std::cmatch m;
  assert(!m.ready());
  assert(m.empty() && m.size() == 0);

  if (std::regex_match("abc", m, re))
  {
    assert(m.ready());
    // Sub-expression 0 is the whole match, then one per mark.
    assert(m.size() == re.mark_count() + 1);
    assert(m[0].matched);
  }
  else
  {
    assert(m.ready());
    assert(m.size() == 0);
  }

  // [re.results.acc]: an index past the end names an unmatched sub_match.
  std::cmatch fresh;
  assert(!fresh[99].matched);
  assert(fresh[99].length() == 0);

  bool caught = false;
  try
  {
    throw std::regex_error(std::regex_constants::error_paren);
  }
  catch (const std::regex_error &e)
  {
    caught = e.code() == std::regex_constants::error_paren;
  }
  assert(caught);

  // Patterns with different counts, so the assertion fails if swap is a no-op.
  std::regex a("(x)"), b("(y)(z)");
  a.swap(b);
  assert(a.mark_count() == 2 && b.mark_count() == 1);

  // '(' inside a bracket expression is a literal, not a group.
  std::regex bracket("[(]x");
  assert(bracket.mark_count() == 0);

  // An unbalanced pattern is rejected. [re.regex.construct]
  bool threw = false;
  try
  {
    std::regex bad("(");
  }
  catch (const std::regex_error &e)
  {
    threw = e.code() == std::regex_constants::error_paren;
  }
  assert(threw);
  return 0;
}
