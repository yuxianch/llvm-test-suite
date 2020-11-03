use ics_subs;

my $cmake_log = "$optset_work_dir/cmake.log";
my $ninja_log = "$optset_work_dir/ninja.log";
my $run_all_lf = "$optset_work_dir/run_all.lf";

my $test_src;
my $test_path;
my %data;
my $build_output;
my $build_dir = "$optset_work_dir/build";
my $lit = "../lit/lit.py";

my $sycl_backend = "";
my $device = "";

sub unxpath
{
    my $fpath;
    $fpath = shift;
    $fpath =~ s/\\/\//g;

    return $fpath;
}

sub getSrc
{
    $test_src = "$ENV{ICS_WSDIR}/llvm/sycl/test";
    if ( -d "llvm/sycl/test" ) {
      $test_src = getcwd()."/llvm/sycl/test";
    }
    elsif (! -d $test_src) {
      my $ics_proj = $ENV{ICS_PROJECT};
      my $ics_ver  = $ENV{ICS_VERSION};
      my $mirror   = unxpath($ENV{ICS_GIT_MIRROR});
      return BADTEST if ! $ics_proj or
                        ! $ics_ver  or
                        ! $mirror;

      my ($llvm) = grep { $_->{DEST} eq "llvm" } get_project_chunks($ics_proj, $ics_ver);

      return BADTEST if ! defined $llvm or
                        ! $llvm->{REPO} or
                        ! $llvm->{REV};

      rmtree("llvm");
      my $shared_opt = is_windows() ? "" : "--shared";
      execute("git clone -n $shared_opt $mirror/$llvm->{REPO} llvm && cd llvm && git checkout $llvm->{REV}");
      return BADTEST if $command_status;

      $test_src = getcwd()."/llvm/sycl/test";
    }
}

sub getList
{
    # for separate build and test sessions it'd be better to store
    # build phase results in some file and then reread the data
    my @list = sort keys %data;

    if (! @list) {
      # test name cannot include '/' or '\', so replace '/' with '~'
      @list = map { s/.*test\///; s/~/~~/g; s/\//~/g; $_ } alloy_find($test_src, '.*\.cpp|.*\.c');
      # exclude files whose path includes "Input"
      my @indexToKeep = grep { $list[$_] !~ /\bInputs\b/ } 0..$#list;
      @list = @list[@indexToKeep];
    }

    return @list;
}

sub getTestPath
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

    return $command_output;
}

sub generate_run_result
{
    my $output = shift;
    my $result = "";
    getTestPath();

    for my $line (split /^/, $output){
      if ($line =~ m/^(.*): SYCL :: ExtraTests\/tests\/\Q$test_path\E \(.*\)/i) {
        $result = $1;
        if ($result =~ m/^PASS/ or $result =~ m/^XFAIL/) {
          # Expected PASS and Expected FAIL
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
      if ($line =~ m/^.*: SYCL :: ExtraTests\/tests\/\Q$test_path\E \(.*\)/i) {
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
    my $c_flags = "$current_optset_opts $compiler_list_options $compiler_list_options_c $opt_c_compiler_flags";
    my $cpp_flags = "$current_optset_opts $compiler_list_options $compiler_list_options_cpp  $opt_cpp_compiler_flags";
    my $c_cmplr = &get_cmplr_cmd('c_compiler');
    my $cpp_cmplr = &get_cmplr_cmd('cpp_compiler');
    my $c_cmd_opts = '';
    my $cpp_cmd_opts = '';
    my $thread_opts = '';

    if ( $cpp_cmplr =~ /([^\s]*)\s(.*)/)
    {
        $cpp_cmplr = $1;
        $cpp_cmd_opts = $2;
        # Do not pass "-c" or "/c" arguments because some commands are executed with onestep
        $cpp_cmd_opts =~ s/[-\/]{1}c//;
    }
    if ($cmplr_platform{OSFamily} eq "Windows") {
        $c_cmplr = "clang-cl";
        if ($cpp_cmplr eq 'clang++') {
            $cpp_cmplr = "clang-cl";
            # Add "/EHsc" for syclos
            $cpp_cmd_opts .= " /EHsc";
        }
    } else {
        $c_cmplr = "clang";
        $thread_opts = "-lpthread";
    }

    my $collect_code_size="Off";
    execute("which llvm-size");
    if ($command_status == 0)
    {
        $collect_code_size="On";
    }

    if ( $current_optset =~ m/ocl/ )
    {
        $sycl_backend = "PI_OPENCL";
    } elsif ( $current_optset =~ m/nv_gpu/ ) {
        $sycl_backend = "PI_CUDA";
    } elsif ( $current_optset =~ m/gpu/ ) {
        $sycl_backend = "PI_LEVEL_ZERO";
    } else {
        $sycl_backend = "PI_OPENCL";
    }

    if ( $current_optset =~ m/opt_use_cpu/ )
    {
        $device = "cpu";
    }elsif ( $current_optset =~ m/opt_use_gpu/ ){
        $device = "gpu";
    }elsif ( $current_optset =~ m/opt_use_acc/ ){
        $device = "acc";
    }elsif ( $current_optset =~ m/opt_use_nv_gpu/ ){
        $device = "gpu";
    }else{
        $device = "host";
    }

    safe_Mkdir($build_dir);
    chdir_log($build_dir);
    execute( "cmake -G Ninja ../"
           . " -DTEST_SUITE_SUBDIRS=SYCL -DTEST_SUITE_LIT=$lit"
           . " -DSYCL_BE=$sycl_backend -DSYCL_TARGET_DEVICES=$device"
           . " -DCMAKE_BUILD_TYPE=None" # to remove predifined options
           . " -DCMAKE_C_COMPILER=\"$c_cmplr\""
           . " -DCMAKE_CXX_COMPILER=\"$cpp_cmplr\""
           . " -DCMAKE_C_FLAGS=\"$c_cmd_opts $c_flags\""
           . " -DCMAKE_CXX_FLAGS=\"$cpp_cmd_opts $cpp_flags\""
           . " -DCMAKE_THREAD_LIBS_INIT=\"$thread_opts\""
           . " -DTEST_SUITE_COLLECT_CODE_SIZE=\"$collect_code_size\""
           . " -DLIT_EXTRA_ENVIRONMENT=\"SYCL_ENABLE_HOST_DEVICE=1\""
           . " -DSYCL_EXTRA_TESTS_SRC=$test_src"
           . " |& tee $cmake_log 2>&1"
    );
    $build_output = $command_output;
}

sub run_build
{
    my $res;

    # run cmake
    run_cmake();
    if (($res = $command_status) != PASS) {
      return $res;
    }

    # run ninja to copy files to SYCL/ExtraTests/tests folder
    execute( "ninja ExtraTests |& tee $ninja_log");
    $build_output .= $command_output;

    my $test_full_path = "$optset_work_dir/SYCL/ExtraTests/tests/" . getTestPath();
    if (! -d $test_full_path and ! -f $test_full_path) {
      $build_output .= "\n$test_full_path not exist!\n";
      $res = COMPFAIL;
    } elsif (($res = $command_status) == PASS) {
      # rename lit.site.cfg.py.in and lit.cfg.py in SYCL/ExtraTests/tests
      my $test_folder = "$optset_work_dir/SYCL/ExtraTests/tests";
      my @lit_files = alloy_find($test_folder, 'lit\.site\.cfg\.py\.in|lit\.cfg\.py');
      foreach my $file (@lit_files) {
        rename($file, "$file.ori");
      }
      # copy lit files in SYCL to SYCL/ExtraTests since some variables in SYCL/ExtraTests are not defined
      copy("$optset_work_dir/SYCL/lit.site.cfg.py.in", "$optset_work_dir/SYCL/ExtraTests/") or die "Copy failed: $!";
      copy("$optset_work_dir/SYCL/lit.cfg.py", "$optset_work_dir/SYCL/ExtraTests/") or die "Copy failed: $!";

    }

    return $res;
}

sub BuildSuite
{
    if (getSrc() eq BADTEST) {
      return BADTEST;
    }

    my @list = getList(@_);

    my $ret = $COMPFAIL;
    $build_output = "";
    $current_test = "";
    my $res = run_build();
    foreach my $tst (@list) {
      $current_test = $tst;
      $data{$tst}{res} = $res;
      if ($res eq PASS) {
        $ret = PASS;
      }
      else {
        $data{$tst}{msg} = "cmake/ninja return non zero";
      }
      $data{$tst}{co} = $build_output || "check cmake or ninja log files\n";
    }
    return $ret; # need to return PASS if at least one test succeeds
}

sub BuildTest
{
    if (getSrc() eq BADTEST) {
      return BADTEST;
    }

    $build_output = "";
    my $ret = $COMPFAIL;
    my $res = run_build();
    if ($res == PASS) {
      $ret = PASS;
    }
    else {
      $failure_message = "cmake/ninja return non zero";
    }
    $compiler_output = $build_output || "check cmake or ninja log files\n";

    return $ret;
}

sub RunSuite
{
    my $ret = PASS;
    my @list = getList(@_);
    my $lscl_output = "";

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
          # show devices info
          $lscl_output = lscl();
          set_tool_path();
          chdir_log($build_dir);
          execute("python3 $lit -a SYCL/ExtraTests > $run_all_lf 2>&1");
        }

        $execution_output .= $lscl_output;

        my $run_output = file2str($run_all_lf);
        $res = generate_run_result($run_output);
        my $filtered_output = generate_run_test_lf($run_output);
        $execution_output .= $filtered_output;

        if ($res != PASS) {
          $msg = $failure_message;
          $ret = RUNFAIL;
        }
      }

      finalize_test($tst,
                    $res,
                    '', # status
                    0, # exesize
                    0, # objsize
                    0, # compile_time
                    0, # link_time
                    0, # execution_time
                    0, # save_time
                    0, # execute_time
                    $msg,
                    0, # total_time
                    $data{$tst}{co},
                    $execution_output
      );
    }
    return $ret; # need to return RUNFAIL if at least one test fails
}

sub RunTest
{
    $execution_output = "";
    getTestPath();
    chdir_log($build_dir);
    # show devices info
    my $lscl_output .= lscl();
    set_tool_path();
    execute("python3 $lit -a SYCL/ExtraTests/tests/$test_path");
    $execution_output = "$lscl_output\n$command_output";
    $failure_message = "test execution exit status $command_status";

    return generate_run_result($command_output);
}

sub set_tool_path
{
    my $tool_path = "";
    if ($cmplr_platform{OSFamily} eq "Windows") {
        $tool_path = "$optset_work_dir/lit/tools/Windows";
    } else {
        $tool_path = "$optset_work_dir/lit/tools/Linux";
    }
    my $env_path = join($path_sep, $tool_path, $ENV{PATH});
    set_envvar("PATH", $env_path, join($path_sep, $tool_path, '$PATH'));
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

1;
