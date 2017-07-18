#!/usr/bin/perl

use subs;
use strict;
use warnings;
use IO::Handle;
use Time::HiRes qw(tv_interval gettimeofday);

# test.pl
#
# runs a test and check its output

sub run($$$) {
  my ($input, $options, $output) = @_;
  my $extraopts = $ENV{'ESBMC_TEST_EXTRA_ARGS'};
  $extraopts = "" unless defined($extraopts);
  my $cmd = "esbmc $extraopts $options $input --constrain-all-states >$output 2>&1";

  print LOG "Running $cmd\n";
  my $tv = [gettimeofday()];
  system $cmd;
  my $elapsed = tv_interval($tv);
  my $exit_value = $? >> 8;
  my $signal_num = $? & 127;
  my $dumped_core = $? & 128;
  my $failed = 0;

  print LOG "  Exit: $exit_value\n";
  print LOG "  Signal: $signal_num\n";
  print LOG "  Core: $dumped_core\n";

  if($signal_num != 0) {
    $failed = 1;
    print "Killed by signal $signal_num";
    if($dumped_core) {
      print " (code dumped)";
    }
  }

  if ($signal_num == 2) {
    # Interrupted by ^C: we should exit too
    print "Halting tests on interrupt";
    exit 1;
  }

  system "echo EXIT=$exit_value >>$output";
  system "echo SIGNAL=$signal_num >>$output";

  return ($failed, $elapsed);
}

sub load($) {
  my ($fname) = @_;

  open FILE, "<$fname";
  my @data = <FILE>;
  close FILE;

  chomp @data;
  return @data;
}

sub test($$) {
  my ($name, $test) = @_;
  my ($input, $options, @results) = load($test);

  my $output = $input;
  $output =~ s/\.c$/.out/;
  $output =~ s/\.cpp$/.out/;

  if($output eq $input) {
    print("Error in test file -- $test\n");
    return 1;
  }

  print LOG "Test '$name'\n";
  print LOG "  Input: $input\n";
  print LOG "  Output: $output\n";
  print LOG "  Options: $options\n";
  print LOG "  Results:\n";
  foreach my $result (@results) {
    print LOG "    $result\n";
  }

  my ($failed, $elapsed);
  ($failed, $elapsed) = run($input, $options, $output);

  if(!$failed) {
    print LOG "Execution [OK]\n";
    my $included = 1;
    foreach my $result (@results) {
      if($result eq "--") {
	$included = !$included;
      } else {
	my $r;
	system "grep '$result' '$output' >/dev/null";
	$r = ($included ? $? != 0 : $? == 0);
	if($r) {
	  print LOG "$result [FAILED]\n";
	  $failed = 1;
	} else {
	  print LOG "$result [OK]\n";
	}
      }
    }
  } else {
    print LOG "Execution [FAILED]\n";
  }

  print LOG "\n";

  return ($failed, $elapsed, $output, @results);
}

sub dirs() {
  my @list;

  opendir CWD, ".";
  @list = grep { !/^\./ && -d "$_" && !/CVS/ } readdir CWD;
  closedir CWD;

  @list = sort @list;

  return @list;
}

if(@ARGV != 0) {
  print "Usage:\n";
  print "  test.pl\n";
  exit 1;
}

open LOG,">tests.log";

print "Loading\n";
my @tests = dirs();
my $count = @tests;
if($count == 1) {
  print "  $count test found\n";
} else {
  print "  $count tests found\n";
}
print "\n";
my $failures = 0;
my $xmloutput =  "";
my $timeaccuml = 0.0;
print "Running tests\n";
foreach my $test (@tests) {
  print "  Running $test";

  chdir $test;
  my ($failed, $elapsed, $outputfile, @resultrexps) = test($test, "test.desc");
  $timeaccuml += $elapsed;
  chdir "..";

  $xmloutput = $xmloutput . "<testcase name=\"$test\" time=\"$elapsed\"";

  if($failed) {
    $failures++;
    print "  [FAILED]";
    $xmloutput = $xmloutput . ">\n";
    $xmloutput = $xmloutput . "  <failure message=\"Test regexes \'@resultrexps\' failed\" type=\"nodescript failure\">\n";

    open(LOGFILE, "<$test/$outputfile") or die "Can't open outputfile $test/$outputfile";
    while (<LOGFILE>) {
      my $lump;
      $lump =  $_;
      $lump =~ s/\&/\&amp;/gm;
      $lump =~ s/"/\&quot;/gm;
      $lump =~ s/</\&lt;/gm;
      $lump =~ s/>/\&gt;/gm;
      $xmloutput = $xmloutput . $lump;
    }
    close(LOGFILE);

    $xmloutput = $xmloutput . "</failure>\n</testcase>";
  } else {
    print "  [OK]";
    $xmloutput = $xmloutput . "/>\n";
  }

  printf(" (%.2f seconds)\n", $elapsed);
}
print "\n";

my $io = IO::Handle->new();
if ($io->fdopen(3, "w")) {
  print $io "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<testsuite failures=\"$failures\" hostname=\"pony.ecs.soton.ac.uk\" name=\"ESBMC single threaded regression tests\" tests=\"$count\" time=\"$timeaccuml\">\n";
  print $io $xmloutput;
  print $io "</testsuite>";
}

if($failures == 0) {
  print "All tests were successful\n";
} else {
  print "Tests failed\n";
  if($failures == 1) {
    print "  $failures of $count test failed\n";
  } else {
    print "  $failures of $count tests failed\n";
  }
}

close LOG;

exit $failures;
