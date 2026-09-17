#ifndef DBOX_PARSEOPTIONS_H

#define DBOX_PARSEOPTIONS_H

#include <string>
#include <util/config/cmdline.h>
#include <util/message/message.h>
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
  /// One line naming what this driver offers for a fuller crash report, or
  /// null. Only the driver knows which options it accepts.
  virtual const char *fatal_signal_advice() const
  {
    return nullptr;
  }
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
