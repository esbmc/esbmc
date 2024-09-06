#ifndef DBOX_PARSEOPTIONS_H

#define DBOX_PARSEOPTIONS_H

#include <string>
#include <util/cmdline.h>
#include <util/message.h>
#include <boost/filesystem.hpp>

class parseoptions_baset
{
public:
  parseoptions_baset(
    const struct group_opt_templ *opts,
    int argc,
    const char **argv);

  cmdlinet cmdline;
  virtual void help();
  virtual int doit() = 0;
  virtual int main();
  virtual ~parseoptions_baset() = default;

  void set_verbosity_msg(VerbosityLevel def = VerbosityLevel::Status);

protected:
  // Path to esbmc binary
  boost::filesystem::path executable_path;

private:
  bool exception_occured;
};

#endif
