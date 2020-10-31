use Cwd qw(cwd);
use File::Basename;

my $cmake_log = "$optset_work_dir/cmake.log";
my $cmake_err = "$optset_work_dir/cmake.err";
# Use all.lf to store standard output of run.
my $run_all_lf = "$optset_work_dir/run_all.lf";

my $cwd = cwd();

my @test_name_list = ();
my $short_test_name;
my $test_info;

my $sycl_backend = "";
my $device = "";

my $build_dir = "";
my $lit = "../lit/lit.py";

sub BuildTest
{
    @test_name_list = get_tests_to_run();
    $test_info = get_info();

    $build_dir = $cwd . "/build";

    if ( ! -f "./build/CMakeCache.txt")
    {
        my ( $status, $output) = run_cmake();
        if ( $status)
        {
            rename($cmake_log, $cmake_err);
        }
    }
    if ( ! -f $cmake_err)
    {
        return $PASS;
    }
    $failure_message = "cmake returned non zero exit code";
    return $COMPFAIL;
}

sub RunTest
{
    my ( $status, $output) = do_run($test_info);
    my $res = "";
    if (-e $run_all_lf)
    {
        my $run_output = file2str("$run_all_lf");
        $res = generate_run_result($run_output);
        my $filtered_output = generate_run_test_lf($run_output);
        $execution_output .= $filtered_output;
    } else {
        $res = generate_run_result($output);
    }
    return $res;
}

sub do_run
{
    my $r = shift;
    my $path = "$r->{fullpath}";

    my $test_number = scalar(@test_name_list);
    if ($test_number > 10) {
      if (! -e $run_all_lf) {
        set_tool_path();
        chdir_log($build_dir);
        execute("python3 $lit -a . > $run_all_lf 2>&1");
        chdir_log($optset_work_dir);
      }
    } else {
      set_tool_path();
      chdir_log($build_dir);
      execute("python3 $lit -a $path");
    }

    $execution_output = "$command_output";
    return $command_status, $command_output;
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

sub get_info
{
    my $test_file = file2str("./config_sycl/$current_test.info");
    $test_file =~ /(.*)\.\bcpp\b/;
    my $path = $1;

    my $short_name = basename( $path);
    $short_test_name = "\Q$short_name";
    $path = dirname( $path);
    my $r = { dir => $path, short_name => $short_name, fullpath => "$path/$short_name.cpp"};

    return $r;
}

sub generate_run_result
{
    my $output = shift;
    my $result = "";
    for my $line (split /^/, $output){
      if ($line =~ m/^(.*): SYCL :: .*\b$short_test_name\b\.cpp \(.*\)/i) {
        $result = $1;
        if ($result =~ m/^PASS/ or $result =~ m/^XFAIL/) {
          # Expected PASS and Expected FAIL
          return $PASS;
        } elsif ($result =~ m/^XPASS/) {
          # Unexpected PASS
          $failure_message = "Unexpected pass";
          return $RUNFAIL;
        } elsif ($result =~ m/^TIMEOUT/) {
          # Exceed test time limit
          $failure_message = "Reached timeout";
          return $RUNFAIL;
        } elsif ($result =~ m/^FAIL/) {
          # Unexpected FAIL
          next;
        } elsif ($result =~ m/^UNSUPPORTED/) {
          # Unsupported tests
          return $SKIP;
        } else {
          # Every test should have result.
          # If not, it is maybe something wrong in processing result
          return $FILTERFAIL;
        }
      }

      if ($result =~ m/^FAIL/) {
        if ($line =~ m/Assertion .* failed/ or $line =~ m/Assertion failed:/) {
          $failure_message = "Assertion failed";
          return $RUNFAIL;
        } elsif ($line =~ m/No device of requested type available/) {
          $failure_message = "No device of requested type available";
          return $RUNFAIL;
        } elsif ($line =~ m/error: CHECK.*: .*/) {
          $failure_message = "Check failed";
          return $RUNFAIL;
        } elsif ($line =~ m/fatal error:.* file not found/) {
          $failure_message = "File not found";
          return $RUNFAIL;
        } elsif ($line =~ m/^note: command had no output/) {
          $failure_message = "command had no output";
          return $RUNFAIL;
        } elsif ($line =~ m/error: command failed with exit status: ([\-]{0,1}[0]{0,1}[x]{0,1}[0-9a-f]{1,})/) {
          $failure_message = "command failed with exit status $1";
          return $RUNFAIL;
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
      if ($line =~ m/^.*: SYCL :: .*\b$short_test_name\b\.cpp \(.*\)/i) {
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

    #if ( get_running_device() == RUNNING_DEVICE_CPU )
    if ( $current_optset =~ m/opt_use_cpu/ )
    {
        $device = "cpu";
    #}elsif ( get_running_device() == RUNNING_DEVICE_GPU ){
    }elsif ( $current_optset =~ m/opt_use_gpu/ ){
        $device = "gpu";
    #}elsif ( get_running_device() == RUNNING_DEVICE_ACC ){
    }elsif ( $current_optset =~ m/opt_use_acc/ ){
        $device = "acc";
    #}elsif ( get_running_device() == RUNNING_DEVICE_NV_GPU ){
    }elsif ( $current_optset =~ m/opt_use_nv_gpu/ ){
        $device = "gpu";
    }else{
        $device = "host";
    }

    safe_Mkdir($build_dir);
    chdir_log($build_dir);
    execute( "cmake -G Ninja ../ -DTEST_SUITE_SUBDIRS=SYCL -DTEST_SUITE_LIT=$lit"
                                          . " -DSYCL_BE=$sycl_backend -DSYCL_TARGET_DEVICES=$device"
                                          . " -DCMAKE_BUILD_TYPE=None" # to remove predifined options
                                          . " -DCMAKE_C_COMPILER=\"$c_cmplr\""
                                          . " -DCMAKE_CXX_COMPILER=\"$cpp_cmplr\""
                                          . " -DCMAKE_C_FLAGS=\"$c_cmd_opts $c_flags\""
                                          . " -DCMAKE_CXX_FLAGS=\"$cpp_cmd_opts $cpp_flags\""
                                          . " -DTEST_SUITE_COLLECT_CODE_SIZE=\"$collect_code_size\""
                                          . " -DLIT_EXTRA_ENVIRONMENT=\"SYCL_ENABLE_HOST_DEVICE=1,OverrideDefaultFP64Settings=1,IGC_EnableDPEmulation=1\""
                                          . " > $cmake_log 2>&1"
                                      );
    return $command_status, $command_output;
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

sub print2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub CleanupTest {
  if ($current_test eq $test_name_list[-1]) {
      rename($run_all_lf, "$run_all_lf.last");
      rename($cmake_err, "$cmake_err.last");
  }
}

1;

