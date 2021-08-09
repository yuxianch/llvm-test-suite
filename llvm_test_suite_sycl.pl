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
my $config_folder = "";
my $subdir = "SYCL";
my $insert_command = "";

my $sycl_backend = "";
my $device = "";

my $build_dir = "";
my $lit = "../lit/lit.py";

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

sub init_test
{
    my $suite_feature = $current_suite;
    $suite_feature =~ s/^llvm_test_suite_//;
    #Remove suffix of suite names if it has
    $suite_feature =~ s/~.*$//;
    $config_folder = 'config_sycl';
    if ($suite_feature ne 'sycl' and $suite_feature ne 'sycl_valgrind')
    {
        $config_folder = $config_folder . '_' . $suite_feature;
        $subdir = uc $suite_feature;
        $subdir = 'SYCL_' . $subdir;

        my $sycl_dir = "./SYCL";
        my @file_list = alloy_find($sycl_dir, '(.*\.(h|hpp|H|HPP)|lit\..*|CMakeLists\.txt)');

        # Copy files to folder $subdir
        foreach my $file (@file_list) {
            $file =~ s/^\.\/SYCL\///;
            my $rel_file_path = dirname($file);
            my $file_path_in_subdir = "$optset_work_dir/$subdir";
            if ($rel_file_path ne '.') {
                $file_path_in_subdir = $file_path_in_subdir . "/$rel_file_path";
            }
            my $file_in_sycl = "$optset_work_dir/SYCL/$file";
            if ( -d $file_path_in_subdir) {
                cp($file_in_sycl, $file_path_in_subdir);
            }
        }

        my $cmake_file = "$subdir/CMakeLists.txt";
        if ( ! -d "$subdir/External") {
            `sed -i '/^add_subdirectory(External)/s/^/#/g' $cmake_file`;
        }
        if ( ! -d "$subdir/ExtraTests"){
            `sed -i '/^add_subdirectory(ExtraTests)/s/^/#/g' $cmake_file`;
        }

    }

    #Remove untested source files from $subdir if it run with several subsuites
    my $info_dir = "$optset_work_dir/$config_folder";
    my @info_files = glob("$info_dir/*.info");

    my @in_test_list = get_test_list();
    my %in_test_hash = map { $_ => 1 } @in_test_list;
    my @outof_test_list = ();

    for my $file (@info_files) {
      $file = basename($file);
      $file =~ s/\.info//;
      if (!exists($in_test_hash{$file})) {
        push(@outof_test_list, $file);
      }
    }
    for my $test (@outof_test_list) {
      my $test_info = get_info($test);
      my $path = "$test_info->{fullpath}";
      rmtree($path);
    }

    #add Valgrind command for test run, added by Xingxu"
    if ($suite_feature eq 'sycl_valgrind') {
      my $valgrind_dir = $cwd . "/valgrind_reports";
      safe_Mkdir($valgrind_dir);
      $insert_command = "/rdrive/ref/valgrind/v3.16.0/efi2/bin/valgrind --leak-check=full --show-leak-kinds=all --trace-children=yes --log-file=$cwd/valgrind_reports/log.%%p";
    }

    return PASS;
}

sub BuildTest
{
    $build_dir = $cwd . "/build";
    safe_Mkdir($build_dir);

    @test_name_list = get_tests_to_run();
    if ($current_test eq $test_name_list[0])
    {
        init_test();
        chdir_log($build_dir);

        my ( $status, $output) = run_cmake();
        if ( $status)
        {
            rename($cmake_log, $cmake_err);
        } else {
            # If there is no configuration issue, print device info
            my $lscl_output = lscl();
            append2file($lscl_output, $cmake_log);
        }
    } else {
      chdir_log($build_dir);
    }

    $compiler_output = file2str($cmake_log);

    if ( ! -f $cmake_err)
    {
        return $PASS;
    }

    $failure_message = "cmake returned non zero exit code";
    return $COMPFAIL;
}

sub RunTest
{
    $test_info = get_info();
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

    if (! -e $run_all_lf) {
      my @whole_suite_test = sort(get_test_list());
      my @current_test_list = sort(@test_name_list);
      my $is_suite = is_same(\@current_test_list, \@whole_suite_test);
      my $python = "python3";
      my $timeset = "";
      my $matrix = "";
      my $jobset = "-j 8";

      if (defined $ENV{'CURRENT_GPU_DEVICE'}) {
        my $current_gpu = $ENV{'CURRENT_GPU_DEVICE'};
        if ($current_gpu =~ m/ats/) {
          $python = "/usr/bin/python3";
          $matrix = "-Dmatrix=1";
          $jobset = "";
        } elsif ($current_gpu =~ m/pvc/) {
          $timeset = "--timeout 1200";
          $jobset = "";
        }
      }

      if ($current_suite eq 'llvm_test_suite_sycl_valgrind'){
        $timeset = "--timeout 0";
      }

      if ($is_suite) {
        set_tool_path();
        execute("$python $lit -a $matrix $jobset . $timeset > $run_all_lf 2>&1");
      } else {
        set_tool_path();
        execute("$python $lit -a $matrix $path $timeset");
      }
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
    my $test_name = shift;
    $test_name = $current_test if ! defined $test_name or $test_name eq "";

    my $test_file = file2str("$optset_work_dir/$config_folder/$test_name.info");
    $short_test_name = $test_file;
    $short_test_name =~ s/^$subdir\///;

    my $short_name = basename($test_file);
    my $path = dirname($test_file);
    my $r = { dir => $path, short_name => $short_name, fullpath => $test_file};

    return $r;
}

sub generate_run_result
{
    my $output = shift;
    my $result = "";
    for my $line (split /^/, $output){
      if ($line =~ m/^(.*): SYCL :: \Q$short_test_name\E \(.*\)/) {
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
          # If not, it is maybe something wrong in processing result or missing result
          $failure_message = "Result not found";
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
        } elsif ($line =~ m/error: command failed with exit status: ([\-]{0,1}[0]{0,1}[x]{0,1}[0-9a-f]{1,})/) {
          $failure_message = "command failed with exit status $1";
          return $RUNFAIL;
        }
      }
    }

    # Every test should have result.
    # If not, it is maybe something wrong in processing result or missing result
    $failure_message = "Result not found";
    return $FILTERFAIL;
}

sub generate_run_test_lf
{
    my $output = shift;
    my $filtered_output = "";

    my $printable = 0;
    for my $line (split /^/, $output) {
      if ($line =~ m/^.*: SYCL :: \Q$short_test_name\E \(.*\)/) {
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
    my $cpp_flags = "$current_optset_opts $compiler_list_options $compiler_list_options_cpp $opt_cpp_compiler_flags";
    my $link_flags = "$linker_list_options $opt_linker_flags";
    my $c_cmplr = &get_cmplr_cmd('c_compiler');
    my $cpp_cmplr = &get_cmplr_cmd('cpp_compiler');
    my $c_cmd_opts = '';
    my $cpp_cmd_opts = '';
    my $thread_opts = '';

    ($c_cmplr, $c_cmd_opts) = remove_opt($c_cmplr);
    ($cpp_cmplr, $cpp_cmd_opts) = remove_opt($cpp_cmplr);
    $c_cmd_opts .= $c_flags;
    $cpp_cmd_opts .= $cpp_flags;

    if ($cmplr_platform{OSFamily} eq "Windows") {
    # Windows
        if ($compiler !~ /xmain/) {
            $c_cmplr = "clang-cl";
            $cpp_cmplr = "clang-cl";
            # Add "/EHsc" for syclos
            $cpp_cmd_opts .= " /EHsc";
        } else {
            $c_cmplr = "clang";
            $cpp_cmplr = 'clang++';
            $c_cmd_opts = convert_opt($c_cmd_opts);
            $cpp_cmd_opts = convert_opt($cpp_cmd_opts);
        }
    } else {
    # Linux
        $c_cmplr = "clang";
        if ($compiler =~ /xmain/) {
            $cpp_cmplr = "clang++";
        }
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

    my $lit_extra_env = "SYCL_ENABLE_HOST_DEVICE=1";
    $lit_extra_env = join_extra_env($lit_extra_env,"CPATH");
    $lit_extra_env = join_extra_env($lit_extra_env,"LIBRARY_PATH");
    $lit_extra_env = join_extra_env($lit_extra_env,"GCOV_PREFIX");
    $lit_extra_env = join_extra_env($lit_extra_env,"GCOV_PREFIX_STRIP");
    $lit_extra_env = join_extra_env($lit_extra_env,"TC_WRAPPER_PATH");

    if ( defined $ENV{PIN_CMD} ) {
        my $pin_cmd = $ENV{PIN_CMD};

        if ($pin_cmd =~ /=/) {
          $lit_extra_env = join(',',$lit_extra_env,$ENV{PIN_CMD});
        } elsif ($insert_command eq "") {
          $insert_command = $pin_cmd;
        }
    }

    if ($insert_command ne "") {
        my $config_file = "$optset_work_dir/SYCL/lit.cfg.py";
        if (! -f $config_file) {
          return COMPFAIL, "File SYCL/lit.cfg.py doesn't exist";
        }

        my $config_file_original = "$config_file.ori";
        # If using tc -rerun, it may repeat inserting so we need to keep the original file and insert on it
        if (! -f $config_file_original) {
          copy($config_file, $config_file_original);
        }

        open my $in, "<", $config_file_original || die "Cannot open file lit.cfg.py.ori: $!";
        open my $out, ">", $config_file || die "Cannot open file lit.cfg.py: $!";
        while (<$in>) {
          s/env\s+SYCL_DEVICE_FILTER=(\S+)/env SYCL_DEVICE_FILTER=$1 $insert_command /g;
          print $out $_;
        }
        close $in;
        close $out;
    }

    execute( "cmake -G Ninja ../ -DTEST_SUITE_SUBDIRS=$subdir -DTEST_SUITE_LIT=$lit"
                                          . " -DSYCL_BE=$sycl_backend -DSYCL_TARGET_DEVICES=$device"
                                          . " -DCMAKE_BUILD_TYPE=None" # to remove predifined options
                                          . " -DCMAKE_C_COMPILER=\"$c_cmplr\""
                                          . " -DCMAKE_CXX_COMPILER=\"$cpp_cmplr\""
                                          . " -DCMAKE_C_FLAGS=\"$c_cmd_opts\""
                                          . " -DCMAKE_CXX_FLAGS=\"$cpp_cmd_opts\""
                                          . " -DCMAKE_EXE_LINKER_FLAGS=\"$link_flags\""
                                          . " -DCMAKE_THREAD_LIBS_INIT=\"$thread_opts\""
                                          . " -DTEST_SUITE_COLLECT_CODE_SIZE=\"$collect_code_size\""
                                          . " -DLIT_EXTRA_ENVIRONMENT=\"$lit_extra_env\""
                                          . " > $cmake_log 2>&1"
                                      );
    return $command_status, $command_output;
}

sub convert_opt
{
    my $opt = shift;

    # Convert options from MSVC format to clang format
    # For other options, keep them the original format
    $opt =~ s/[\/\-]Od/-O0/g;
    $opt =~ s/[\/]O([0-3]{1})/-O$1/g;
    $opt =~ s/\/Zi/-g/g;
    return $opt;
}

sub remove_opt
{
    my $cmplr_info = shift;

    my $cmplr = '';
    my $cmd_opts = '';
    if ( $cmplr_info =~ /([^\s]*)\s(.*)/)
    {
        $cmplr = $1;
        $cmd_opts = $2;
        chomp $cmd_opts;
        # Do not pass "-c" or "/c" arguments because some commands are executed with onestep
        $cmd_opts =~ s/[-\/]{1}c$|[-\/]{1}c\s{1,}//;
        # Do not pass "-fsycl" because it's included in the RUN commands
        $cmd_opts =~ s/-fsycl$|-fsycl\s{1,}//;
        # Do not pass "-fsycl-unnamed-lambda"
        $cmd_opts =~ s/-fsycl-unnamed-lambda$|-fsycl-unnamed-lambda\s{1,}//;
        # Remove "/EHsc" since it's not supported by clang/clang++
        $cmd_opts =~ s/\/EHsc$|\/EHsc\s{1,}//;
    } else {
        $cmplr = $cmplr_info;
    }
    return $cmplr, $cmd_opts;
}

sub is_same
{
    my($array1, $array2) = @_;

    # Return 0 if two arrays are not the same length
    return 0 if scalar(@$array1) != scalar(@$array2);

    for(my $i = 0; $i <= $#$array1; $i++) {
        if ($array1->[$i] ne $array2->[$i]) {
           return 0;
        }
    }
    return 1;
}

sub join_extra_env
{
    my $extra_env = shift;
    my $env_var = shift;

    my $env = '';
    if (defined $ENV{$env_var}) {
        $env = "$env_var=$ENV{$env_var}";
        $extra_env = join(',',$extra_env,$env);
    }

    return $extra_env;
}

sub file2str
{
    my $file = shift;
    ###
    local $/=undef;
    open FD, "<$file" or die "Fail to open file $file!\n";
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

sub append2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">>$file" or die("ERROR: Failed to open $file for write.");

    my $last = '';
    while(<FD>) {
        if ($_ =~ /\Q$s\E/) {
            $last = $_;
            last;
        }
    }
    if ($last eq '') {
        print FD $s;
    }

    close FD;
}

sub CleanupTest {
  if ($current_test eq $test_name_list[-1]) {
      rename($run_all_lf, "$run_all_lf.last");
      rename($cmake_err, "$cmake_err.last");
  }
}

1;

