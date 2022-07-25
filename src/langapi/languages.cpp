#include <langapi/languages.h>
#include <langapi/mode.h>

languagest::languagest(const namespacet &_ns, language_idt lang) : ns(_ns)
{
  language = new_language(lang);
}

languagest::~languagest()
{
  delete language;
}
