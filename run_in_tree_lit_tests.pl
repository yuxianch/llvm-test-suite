use ics_subs;

my $run_all_lf = "$optset_work_dir/run_all.lf";
my $wsdir;
my $tests_abspath;
my $test_path;
my %data;
my $build_output;
my $build_dir = "$optset_work_dir/build";
my $cmplr_root;
my $sycl_backend = "";
my $ics_os = is_windows() ? "Windows":"Linux";
my $exe_postfix = is_windows() ? ".exe":"";

sub unxpath
{
    my $fpath;
    $fpath = shift;
    $fpath =~ s/\\/\//g;

    return $fpath;
}

sub get_src
{
    my $sycl_src = "llvm/sycl/test/on-device";
    my $cwd = getcwd();
    if ( -d $sycl_src ) {
      $tests_abspath = "$cwd/$sycl_src";
      $wsdir = "$cwd";
    }
    elsif (! -d "$ENV{ICS_WSDIR}/$sycl_src") {
      my $ics_proj = $ENV{ICS_PROJECT};
      my $ics_ver  = $ENV{ICS_VERSION};
      my $mirror   = unxpath($ENV{ICS_GIT_MIRROR});
      return BADTEST if ! $ics_proj or
                        ! $ics_ver  or
                        ! $mirror;

      my $co_id = "";
      my $llvm = "";
      if ($ics_proj eq "syclos") {
        ($llvm) = grep { $_->{DEST} eq "llvm" } get_project_chunks("xmain");
        return BADTEST if ! defined $llvm or
                          ! $llvm->{REPO};

        my ($tag) = split "_", $ics_ver;
        $co_id = "sycl-nightly/$tag";
      } elsif ($ics_proj =~ /xmain/) {
        ($llvm) = grep { $_->{DEST} eq "llvm" } get_project_chunks($ics_proj, $ics_ver);

        return BADTEST if ! defined $llvm or
                          ! $llvm->{REPO} or
                          ! $llvm->{REV};

        $co_id = "$llvm->{REV}";
      } else {
        # Unsupported for other unknown project
        return BADTEST;
      }

      my $shared_opt = "";
      if (is_windows()) {
        execute("git --version");
        # git 2.10.1 has a bug in using "--shared"
        if ($command_output !~ /^git version 2.10.1/) {
          $shared_opt = "--shared";
        }
      }

      rmtree("llvm");
      execute("git clone -n $shared_opt $mirror/$llvm->{REPO} llvm && cd llvm && git checkout $co_id");
      return BADTEST if $command_status;

      $tests_abspath = "$cwd/$sycl_src";
      $wsdir = "$cwd";
    } else {
      $tests_abspath = "$ENV{ICS_WSDIR}/$sycl_src";
      $wsdir = "$ENV{ICS_WSDIR}";
    }
}

sub get_list
{
    # for separate build and test sessions it'd be better to store
    # build phase results in some file and then reread the data
    my @list = sort keys %data;

    if (! @list) {
      # test name cannot include '/' or '\', so replace '/' with '~'
      @list = map { s/.*test\/on-device\///; s/~/~~/g; s/\//~/g; $_ } alloy_find("$tests_abspath", '.*\.cpp|.*\.c');
      # exclude files whose path includes "Input"
      my @indexToKeep = grep { $list[$_] !~ /\bInputs\b/ } 0..$#list;
      @list = @list[@indexToKeep];
    }

    return @list;
}

sub get_test_path
{
    if ($current_test ne "") {
      $test_path = $current_test;
      $test_path =~ s/~~/\\/g;        # replace: '~~' -> '\'
      $test_path =~ s/~([^~])/\/$1/g; # replace: '~' -> '/'
      $test_path =~ s/\\/~/g;          # replace back: '\' -> '~'
    } else {
      $test_path = "";
    }
    return $test_path;
}

sub report_result
{
    my $testname = shift;
    my $result = shift;
    my $message = shift;
    my $comp_output = shift;
    my $exec_output = shift;

    finalize_test($testname,
                  $result,
                  '', # status
                  0, # exesize
                  0, # objsize
                  0, # compile_time
                  0, # link_time
                  0, # execution_time
                  0, # save_time
                  0, # execute_time
                  $message,
                  0, # total_time
                  $comp_output,
                  $exec_output
    );
}

sub lscl {
    my $args = shift;

    my $os_flavor = is_windows() ? "win.x64" : "lin.x64";

    my $lscl_bin = $ENV{"ICS_TESTDATA"} .
      "/mainline/CT-SpecialTests/opencl/tools/$os_flavor/bin/lscl";

    my @cmd = ($lscl_bin);

    push(@cmd, @{$args}) if $args;

    # lscl show warning "clGetDeviceInfo failed: Invalid value"
    # which contains string "fail", cause tc fail
    # Remove OS type check because RHEL8 has the same issue
    push(@cmd, "--quiet");

    execute(join(" ", @cmd));

    my $output = "\n  ------ lscl output ------\n"
               . "$command_output\n";

    return $output;
}

sub check_device
{
    my $dev = shift;
    my $lscl_output = shift;

    my $status = PASS;
    my $message = "";
    if ($lscl_output !~ /\b$dev\b/) {
      $status = RUNFAIL;
      $message = "lscl: No $dev found";
    }
    return ($status, $message);
}

sub generate_run_result
{
    my $output = shift;
    my $result = "";
    get_test_path();

    for my $line (split /^/, $output){
      if ($line =~ m/^(.*): SYCL-on-device :: \Q$test_path\E \(.*\)/i) {
        $result = $1;
        if ($result =~ m/^PASS/ or $result =~ m/^XFAIL/) {
          # Expected PASS and Expected FAIL
          $failure_message = "";
          return PASS;
        } elsif ($result =~ m/^XPASS/) {
          # Unexpected PASS
          $failure_message = "Unexpected pass";
          return RUNFAIL;
        } elsif ($result =~ m/^TIMEOUT/) {
          # Exceed test time limit
          $failure_message = "Reached timeout";
          return RUNFAIL;
        } elsif ($result =~ m/^FAIL/) {
          # Unexpected FAIL
          next;
        } elsif ($result =~ m/^UNSUPPORTED/) {
          # Unsupported tests
          return SKIP;
        } else {
          # Every test should have result.
          # If not, it is maybe something wrong in processing result
          return $FILTERFAIL;
        }
      }

      if ($result =~ /^FAIL/) {
        if ($line =~ /Assertion .* failed/ or $line =~ m/Assertion failed:/) {
          $failure_message = "Assertion failed";
          return RUNFAIL;
        } elsif ($line =~ /No device of requested type available/) {
          $failure_message = "No device of requested type available";
          return RUNFAIL;
        } elsif ($line =~ /python: can't open file /) {
          $failure_message = "lit file issue";
          return RUNFAIL;
        } elsif ($line =~ /\berror\b: (.*)/) {
          my $error_msg = $1;
          # replace some special characters with space
          $error_msg =~ tr/`<>'"/ /;
          $failure_message = $error_msg;
          return RUNFAIL;
        }
      }
    }

    # Every test should have result.
    # If not, it is maybe something wrong in processing result
    return $FILTERFAIL;
}

sub generate_run_test_lf
{
    my $output = shift;
    my $filtered_output = "";

    my $printable = 0;
    for my $line (split /^/, $output) {
      if ($line =~ m/^.*: SYCL-on-device :: \Q$test_path\E \(.*\)/i) {
        $printable = 1;
        $filtered_output .= $line;
        next;
      }

      if ($printable == 1) {
        if ($line =~ m/^[*]{20}/ and length($line) <= 22) {
          $filtered_output .= $line;
          $printable = 0;
          last;
        } else {
          $filtered_output .= $line;
        }
      }
    }
    return $filtered_output;
}

sub run_cmake
{
    my $cmdl = "cmake -G Ninja"
             . " -DLLVM_TARGETS_TO_BUILD=\"X86\""
             . " -DLLVM_EXTERNAL_PROJECTS=\"sycl-test\""
             . " -DLLVM_EXTERNAL_SYCL_TEST_SOURCE_DIR=\"$wsdir/llvm/sycl/test/on-device\""
             . " -DSYCL_SOURCE_DIR=\"$wsdir/llvm/sycl\""
             . " -DOpenCL_LIBRARIES=\"$cmplr_root/lib\""
             . " -DLLVMGenXIntrinsics_SOURCE_DIR=\"$optset_work_dir/vc-intrinsics\""
             . " $wsdir/llvm/llvm";

    execute($cmdl);
    $build_output .= "\n  ------ cmake output ------\n"
                   . "$command_output\n";
}

sub run_build
{
    my $cpp_cmplr = &get_cmplr_cmd('cpp_compiler');
    $cpp_cmplr =~ s/^([^\s]{1,})\s+.*$/$1/;
    $cmplr_root = which($cpp_cmplr);
    $cmplr_root = dirname(dirname($cmplr_root));
    $cmplr_root = unxpath($cmplr_root);

    if ( $current_optset =~ m/ocl/ )
    {
        $sycl_backend = "opencl";
    } elsif ( $current_optset =~ m/nv_gpu/ ) {
        $sycl_backend = "cuda";
    } elsif ( $current_optset =~ m/gpu/ ) {
        $sycl_backend = "level_zero";
    } else {
        $sycl_backend = "opencl";
    }

    set_envvar("SYCL_LIT_USE_HOST_ONLY", 0);

    my $res = PASS;

    safe_Mkdir($build_dir);
    chdir_log($build_dir);

    my $error_msg = "";
    # run cmake
    run_cmake();

    if (($res = $command_status) != PASS) {
      $error_msg = "cmake returned non zero exit code";
      return ($res, $error_msg);
    }

    return ($res, $error_msg);
}

sub run_lit_tests
{
    my $tests_path = shift;

    my $lit = is_windows() ? "./bin/llvm-lit.py":"./bin/llvm-lit";
    my $cmplr_bin_path = "$cmplr_root/bin";
    my $cmplr_lib_path = "$cmplr_root/lib";
    my $cmplr_include_path = "$cmplr_root/include/sycl";
    my $tool_path = "$optset_work_dir/tools/$ics_os";
    my $get_device_tool_path = "$tool_path/get_device_count_by_type$exe_postfix";

    my $l0_header_path = "";
    if (defined $ENV{L0LOADERROOT} and defined $ENV{L0LOADERVER}) {
       $l0_header_path = "$ENV{L0LOADERROOT}/$ENV{L0LOADERVER}/include";
    }

    my $env_path = join($path_sep, $tool_path, $ENV{PATH});
    set_envvar("PATH", $env_path, join($path_sep, $tool_path, '$PATH'));

    my $cmdl = "python3 $lit -a"
             . " --param SYCL_PLUGIN=$sycl_backend"
             . " --param SYCL_TOOLS_DIR=\"$cmplr_bin_path\""
             . " --param SYCL_INCLUDE=\"$cmplr_include_path\""
             . " --param SYCL_LIBS_DIR=\"$cmplr_lib_path\""
             . " --param GET_DEVICE_TOOL=\"$get_device_tool_path\""
             . " --param LEVEL_ZERO_INCLUDE_DIR=\"$l0_header_path\""
             . " $tests_path";

    execute($cmdl);
    return $command_output;
}

sub BuildSuite
{
    if (get_src() eq BADTEST) {
      return BADTEST;
    }

    my @list = get_list(@_);

    $build_output = "";
    $current_test = "";

    # Show devices info
    my $lscl_output = lscl();
    $build_output .= $lscl_output;

    my ($res, $err_msg) = run_build();

    # If a device is missing on the machine, report the issue
    if ($current_optset =~ /gpu/) {
      my ($status, $message) = check_device("GPU", $lscl_output);
      report_result("check_GPU", $status, $message, "", $lscl_output) if $status != PASS;
    }

    if ($sycl_backend eq "opencl") {
      my ($status, $message) = check_device("CPU", $lscl_output);
      report_result("check_CPU", $status, $message, "", $lscl_output) if $status != PASS;
      ($status, $message) = check_device("accelerator", $lscl_output);
      report_result("check_accelerator", $status, $message, "", $lscl_output) if $status != PASS;
    }

    my $ret = COMPFAIL;
    foreach my $tst (@list) {
      $current_test = $tst;
      $data{$tst}{res} = $res;
      $data{$tst}{co} = $build_output;
      if ($res == PASS) {
        $ret = PASS;
      }
      else {
        $data{$tst}{msg} = $err_msg;
      }
    }

    # If all the tests are failed, need to report their report here
    # because RunSuite won't be run
    if ($ret != PASS) {
      foreach my $tst (@list) {
        report_result($tst, COMPFAIL, $err_msg, $build_output, "");
      }
    }

    return $ret; # need to return PASS if at least one test succeeds
}

sub BuildTest
{
    $build_output = "";

    # Show devices info
    my $lscl_output = lscl();

    if ($current_test =~ /^check_(GPU)$/ or $current_test =~ /^check_(CPU)$/ or $current_test =~ /^check_(accelerator)$/) {
      my $dev = $1;
      my ($status, $message) = check_device($dev, $lscl_output);
      $failure_message = $message if $status != PASS;
      $compiler_output .= $lscl_output;
      return $status;
    }

    if (get_src() eq BADTEST) {
      return BADTEST;
    }

    $build_output .= $lscl_output;

    my $ret = $COMPFAIL;
    my ($res, $err_msg) = run_build();
    if ($res == PASS) {
      $ret = PASS;
    }
    else {
      $failure_message = $err_msg;
    }
    $compiler_output = $build_output;

    return $ret;
}

sub RunSuite
{
    my $ret = PASS;
    my @list = get_list(@_);
    my $run_output = "";

    foreach my $tst (@list) {
      $current_test = $tst;
      my $res = $data{$tst}{res};
      my $msg = $data{$tst}{msg} || "";
      if (! defined $res) {
        $res = BADTEST;
        $msg = "no data";
      }

      if ($res == PASS) {
        $execution_output = "";
        if (! -e $run_all_lf) {
          $run_output = run_lit_tests($tests_abspath);
          print2file($run_output, $run_all_lf);
        } else {
          $run_output = file2str($run_all_lf);
        }

        $res = generate_run_result($run_output);
        my $filtered_output = generate_run_test_lf($run_output);
        $execution_output .= "\n  ------ llvm-lit output ------\n"
                           . "$filtered_output\n";

        if ($res != PASS) {
          $msg = $failure_message;
          $ret = RUNFAIL;
        }
      } else {
        $ret = RUNFAIL;
      }

      report_result($tst, $res, $msg, $data{$tst}{co}, $execution_output);
    }

    clean_suite();
    return $ret; # need to return RUNFAIL if at least one test fails
}

sub RunTest
{
    $execution_output = "";

    # For tests check_*, they are not real tests.
    # If RunTest() is called, it means that BuildTest() is passed
    # so we just need to return PASS.
    if ($current_test =~ /^check_(GPU)$/ or $current_test =~ /^check_(CPU)$/ or $current_test =~ /^check_(accelerator)$/) {
      return PASS;
    }

    get_test_path();
    my $run_output = run_lit_tests("$tests_abspath/$test_path");
    $execution_output .= "\n  ------ llvm-lit output ------\n"
                       . "$run_output\n";
    $failure_message = "test execution exit status $command_status";

    return generate_run_result($command_output);
}

sub print2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub file2str
{
    my $file = shift;
    ###
    local $/=undef;
    open FD, "<$file";
    binmode FD;
    my $str = <FD>;
    close FD;
    return $str;
}

sub clean_suite
{
    rename($run_all_lf, "$run_all_lf.last");
}

1;
