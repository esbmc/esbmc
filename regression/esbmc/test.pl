#!/usr/bin/perl

use subs;
use strict;
use warnings;

# test.pl
#
# runs a test and check its output

sub run($$$) {
  my ($input, $options, $output) = @_;
  my $cmd = "esbmc $EXTRAESBMCOPTS $options $input >$output 2>&1";

  print LOG "Running $cmd\n";
  system $cmd;
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

  return $failed;
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

  my $failed = run($input, $options, $output);

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

  return $failed;
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
print "Running tests\n";
foreach my $test (@tests) {
  print "  Running $test";

  chdir $test;
  my $failed = test($test, "test.desc");
  chdir "..";

  if($failed) {
    $failures++;
    print "  [FAILED]\n";
  } else {
    print "  [OK]\n";
  }
}
print "\n";

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
