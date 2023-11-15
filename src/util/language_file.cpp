#include <fstream>
#include <util/language.h>
#include <util/language_file.h>
#include <util/message.h>
#include <util/std_types.h>

language_filet::~language_filet()
{
  if(language != nullptr)
    delete language;
}

void language_filet::get_modules()
{
  language->modules_provided(modules);
}

void language_filest::show_parse(std::ostream &out)
{
  for(auto &it : filemap)
    it.second.language->show_parse(out);
}

bool language_filest::parse()
{
  for(auto &it : filemap)
  {
    // Check that file exists

    std::ifstream infile(it.first.c_str());

    if(!infile)
    {
      log_error("Failed to open {}", it.first);
      return true;
    }

    // parse it

    languaget &language = *(it.second.language);

    if(language.parse(it.first))
    {
      log_error("Parsing of {} failed", it.first);
      return true;
    }

    // what is provided?

    it.second.get_modules();
  }

  return false;
}

bool language_filest::typecheck(contextt &context)
{
// typecheck interfaces
#if 0
  for(filemapt::iterator it=filemap.begin();
      it!=filemap.end(); it++)
  {
    if(it->second.language->interfaces(context))
      return true;
  }
#endif
  // build module map

  for(auto &fm_it : filemap)
  {
    std::set<std::string> &modules = fm_it.second.modules;

    for(const auto &mo_it : modules)
    {
      language_modulet module;
      module.file = &fm_it.second;
      module.name = mo_it;
      modulemap.insert(
        std::pair<std::string, language_modulet>(module.name, module));
    }
  }

  // typecheck files

  for(auto &it : filemap)
  {
    if(it.second.modules.empty())
      if(it.second.language->typecheck(context, ""))
        return true;
  }

  // typecheck modules

  for(auto &it : modulemap)
  {
    if(typecheck_module(context, it.second))
      return true;
  }

  return false;
}

bool language_filest::final(contextt &context)
{
#if 1
  std::set<std::string> languages;

  for(auto &it : filemap)
  {
    if(languages.insert(it.second.language->id()).second)
      if(it.second.language->final(context))
        return true;
  }
#endif

  return false;
}

bool language_filest::interfaces(contextt &)
{
#if 0
  for(filemapt::iterator it=filemap.begin();
      it!=filemap.end(); it++)
  {
    if(it->second.language->interfaces(context))
      return true;
  }
#endif
  return false;
}

bool language_filest::typecheck_module(
  contextt &context,
  const std::string &module)
{
  // check module map

  modulemapt::iterator it = modulemap.find(module);

  if(it == modulemap.end())
  {
    log_error("found no file that provides module {}", module);
    return true;
  }

  return typecheck_module(context, it->second);
}

bool language_filest::typecheck_module(
  contextt &context,
  language_modulet &module)
{
  // already typechecked?

  if(module.type_checked)
    return false;

  // already in progress?

  if(module.in_progress)
  {
    log_error("circular dependency in {}", module.name);
    return true;
  }

  module.in_progress = true;

  // first get dependencies of current module

  std::set<std::string> dependency_set;

  //module.file->language->dependencies();

  for(const auto &it : dependency_set)
  {
    if(typecheck_module(context, it))
    {
      module.in_progress = false;
      return true;
    }
  }

  // type check it

  log_status("Type-checking {}", module.name);

  if(module.file->language->typecheck(context, module.name))
  {
    module.in_progress = false;
    return true;
  }

  module.type_checked = true;
  module.in_progress = false;

  return false;
}
