#!/usr/bin/perl
# test.pl
#
# runs a test and check its output

use subs;
use strict;
#use warnings;
no warnings;
use XML::LibXML;
use vars qw(%tags);

#------------------------------------------------------
# Generate the xml file with result the tests 
my $doc = XML::LibXML::Document->new('1.0', 'utf-8');
my $root = $doc->createElement("all-results");
# Insert time and date
my $time_and_date_run_test = "date_time_run";	
my $second_root = $doc->createElement( $time_and_date_run_test );	#add tag
$root->addChild( $second_root ); #add root
#insert value in the node
#$second_root->appendTextNode("Sep 13, 2012 13:54:35");
$second_root->appendTextNode(get_date_time());
$root->appendChild($second_root);
#------------------------------------------------------

my $llvm = 0;
my $testdesc = "test.desc";


# ---------------------------------------
# run esbmc
sub run($$$) {
	my ($input, $options, $output) = @_;
	
	# for XML file
	#------------------------------
	$tags{"item_04_esbmc-option"} = "esbmc"." ".$input." ".$options;
	#------------------------------
	
	my $cmd;

	if($llvm != 1) {
		$cmd = "esbmc $input $options >$output 2>&1";
	} else {
		$cmd = "llbmc $input $options >$output 2>&1";
	}
	
	print LOG "Running $cmd\n";

	system $cmd;

	my $exit_value = $? >> 8;
	my $signal_num = $? & 127;
	my $dumped_core = $? & 128;
	my $failed = 0;
	
	# for XML file
	#------------------------------
	$tags{"item_06_exit-signal"} = $exit_value;
	$tags{"item_07_signal-num"} = $signal_num;
	$tags{"item_08_core-signal"} = $dumped_core;
	#------------------------------

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
	my @data = ();

	### old read
	# open FILE, "<$fname";
	# my @data = <FILE>;
	# close FILE;
	### old read

	# Parse to xml
	my $parser = XML::LibXML->new;
	my $doc    = $parser->parse_file($fname);	
	my $ter = "";

	for my $item ($doc->findnodes('//test-case')) {	      
		push(@data,$item->findvalue('item_04_file_C_to_test'));    
		push(@data,$item->findvalue('item_05_option_to_run_esbmc'));    
		push(@data,$item->findvalue('item_06_expected_result'));
		# for XML file
		#------------------------------
		$tags{"item_11_summary_test_case"} = $item->findvalue('item_03_summary');
		#------------------------------
		
	}	
	
	

	chomp @data;  
	return @data;
}

sub test($$) {
	my ($name, $test) = @_;

	my ($input, $options, @results) = load($test);

	my $output = $input;
        $output =~ s/\.bc.*$/.out/;
	$output =~ s/\.cpp.*$/.out/;
	$output =~ s/\.c.*$/.out/;

	if($output eq $input) {
		print("Error in test file -- $test\n");
		return 1;
	}
	
	# for XML file
	#------------------------------
	$tags{"item_01_test-name"} = $name;
	$tags{"item_02_input-code"} = $input;
	$tags{"item_03_output-code"} = $output;	
	#------------------------------

	print LOG "Test '$name'\n";
	print LOG "  Input: $input\n";
	print LOG "  Output: $output\n";
	print LOG "  Options: $options\n";
	print LOG "  Results:\n";
	
	my $result;
	foreach my $result (@results) {
		print LOG "    $result\n";
		# for XML file	
		#------------------------------
		if($result =~ m/SUCCESSFUL/){			
			$tags{"item_09_expected-result"} = "[SUCCESSFUL]";
		}else{			
			$tags{"item_09_expected-result"} = "[FAILED]";
		}		
		#------------------------------
		
	}
	

	my $failed = run($input, $options, $output);

	if(!$failed) {
		# for XML file
		#------------------------------
		$tags{"item_05_execution-status"} = "[OKAY]";
		#------------------------------
		
		print LOG "Execution [OK]\n";
		my $included = 1;
		foreach my $result (@results) {			
			if($result eq "--") {
				$included = !$included;
			} else {
				my $r;				
				system "grep '$result' '$output' >/dev/null";
				
				# for XML file
				#------------------------------
				if($failed){
					$tags{"item_10_actual-result"} = "[FAILED]";
				}else{
					$tags{"item_10_actual-result"} = "[SUCCESSFUL]";
				}
				#------------------------------					
				
				$r = ($included ? $? != 0 : $? == 0);				
				if($r) {					
					# expected result is NOT egual to actual result
					print LOG "$result [FAILED]\n";					
					$failed = 1;
					
					# for XML file	
					#------------------------------
					if(not($result =~ m/SUCCESSFUL/)){			
						$tags{"item_10_actual-result"} = "[SUCCESSFUL]";
					}else{			
						$tags{"item_10_actual-result"} = "[FAILED]";
					}		
					#------------------------------					
				} else {
					print LOG "$result [OK]\n";
					
					# expected result is egual to actual result
					# for XML file	
					#------------------------------
					if($result =~ m/SUCCESSFUL/){			
						$tags{"item_10_actual-result"} = "[SUCCESSFUL]";
					}else{			
						$tags{"item_10_actual-result"} = "[FAILED]";
					}		
					#------------------------------					
				}
			}
		}
	} else {
		# for XML file
		#------------------------------
		$tags{"item_05_execution-status"} = "[FAILED]";
		#------------------------------
		print LOG "Execution [FAILED]\n";
	}

	print LOG "\n";	
	
	# for XML -> generate all nodes on xml structure
	#------------------------------------------------------
	my $name_xml = "";
	my $tag = "";
	my $value = "";
	
	# Generate node sort
	#$count_test_run = $count_test_run + 1;
	#my $node_name = "run-test_ID_".$count_test_run;
	
	my $node_name = "run-test";
	my $third_root = $doc->createElement( $node_name );
	$root->addChild( $third_root );
	
	
	for $name_xml (sort(keys %tags)) {
		
		$tag = $doc->createElement($name_xml);		
		$value = $tags{$name_xml};
		$tag->appendTextNode($value);
		$third_root->appendChild($tag);		
	}	
	
	#------------------------------------------------------
	
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

#get date and time 
sub get_date_time(){
	my $time = time;    # or any other epoch timestamp
	my $get_time_date = "";
	my @months = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec");
	my ($sec, $min, $hour, $day,$month,$year) = (localtime($time))[0,1,2,3,4,5];
	# You can use 'gmtime' for GMT/UTC dates instead of 'localtime'
	$get_time_date = $months[$month]." ".$day.", ".($year+1900)." ".$hour.":".$min.":".$sec;	
	return $get_time_date;
}
# End functions
# -------------------------------------------------------------------


# Main
# ------------------------------------------------------------------
if(@ARGV != 0) {
	if ($ARGV[0] eq "--llvm") {
	 $llvm = 1;
	 $testdesc = "testllvm.desc";
	} else {
	print "Usage:\n";
	print "  test.pl\n";
	print "  test.pl --llvm\n";
	exit 1;
	}
} 


#Log run
open LOG,">tests.log";
#--------

#Loading test cases
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
					# PATH, file.desc
	my $failed = test($test, $testdesc);
	#
	chdir "..";

	#count fails
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

#--------
close LOG;

#--------------------------------------------------
# close xml file
$doc->setDocumentElement($root);
# Save the resulting XML file
open(FILE_XML , ">test_log.xml") or die "Could not possible to write xml log file!";
print FILE_XML $doc->toString(1);
close FILE_XML;
