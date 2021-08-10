// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: The test irregularly reports incorrect.
// UNSUPPORTED: TEMPORARY_DISABLED

// This test checks handling of parallel_for() accepting nd_range and
// two or more reductions.

#include "reduction_utils.hpp"

#include <CL/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>

using namespace cl::sycl;

template <typename... Ts> class KNameGroup;
template <typename T, bool B> class KName;

constexpr access::mode RW = access::mode::read_write;
constexpr access::mode DW = access::mode::discard_write;

template <typename T>
bool cherkResultIsExpected(int TestCaseNum, T Expected, T Computed,
                           bool IsSYCL2020) {
  bool Success;
  if (!std::is_floating_point<T>::value)
    Success = (Expected == Computed);
  else
    Success = std::abs((Expected / Computed) - 1) < 0.5;

  if (!Success) {
    std::cerr << "Is SYCL2020 mode: " << IsSYCL2020 << std::endl;
    std::cerr << TestCaseNum << ": Expected value = " << Expected
              << ", Computed value = " << Computed << "\n";
  }

  return Success;
}

// Returns 0 if the test case passed. Otherwise, some non-zero value.
template <class Name, bool IsSYCL2020, typename T1, access::mode Mode1,
          typename T2, access::mode Mode2, typename T3, access::mode Mode3,
          typename T4, access::mode Mode4, class BinaryOperation1,
          class BinaryOperation2, class BinaryOperation3,
          class BinaryOperation4>
int testOne(queue &Q, T1 IdentityVal1, T1 InitVal1, BinaryOperation1 BOp1,
            T2 IdentityVal2, T2 InitVal2, BinaryOperation2 BOp2,
            T3 IdentityVal3, T3 InitVal3, BinaryOperation3 BOp3,
            T4 IdentityVal4, T3 InitVal4, BinaryOperation4 BOp4,
            usm::alloc AllocType4, size_t NWorkItems, size_t WGSize) {
  buffer<T1, 1> InBuf1(NWorkItems);
  buffer<T2, 1> InBuf2(NWorkItems);
  buffer<T3, 1> InBuf3(NWorkItems);
  buffer<T4, 1> InBuf4(NWorkItems);
  buffer<T1, 1> OutBuf1(1);
  buffer<T2, 1> OutBuf2(1);
  buffer<T3, 1> OutBuf3(1);

  auto Dev = Q.get_device();
  if (AllocType4 == usm::alloc::shared &&
      !Dev.get_info<info::device::usm_shared_allocations>())
    return 0;
  if (AllocType4 == usm::alloc::host &&
      !Dev.get_info<info::device::usm_host_allocations>())
    return 0;
  if (AllocType4 == usm::alloc::device &&
      !Dev.get_info<info::device::usm_device_allocations>())
    return 0;
  T4 *Out4 = (T4 *)malloc(sizeof(T4), Dev, Q.get_context(), AllocType4);
  if (Out4 == nullptr)
    return 1;

  // Initialize the arrays with sentinel values
  // and pre-compute the expected result 'CorrectOut'.
  T1 CorrectOut1;
  T2 CorrectOut2;
  T3 CorrectOut3;
  T4 CorrectOut4;
  initInputData(InBuf1, CorrectOut1, IdentityVal1, BOp1, NWorkItems);
  initInputData(InBuf2, CorrectOut2, IdentityVal2, BOp2, NWorkItems);
  initInputData(InBuf3, CorrectOut3, IdentityVal3, BOp3, NWorkItems);
  initInputData(InBuf4, CorrectOut4, IdentityVal4, BOp4, NWorkItems);

  if (Mode1 == access::mode::read_write)
    CorrectOut1 = BOp1(CorrectOut1, InitVal1);
  if (Mode2 == access::mode::read_write)
    CorrectOut2 = BOp2(CorrectOut2, InitVal2);
  if (Mode3 == access::mode::read_write)
    CorrectOut3 = BOp3(CorrectOut3, InitVal3);
  // discard_write mode for USM reductions is available only SYCL2020.
  if (Mode4 == access::mode::read_write || !IsSYCL2020)
    CorrectOut4 = BOp4(CorrectOut4, InitVal4);

  // Inititialize data.
  {
    auto Out1 = OutBuf1.template get_access<access::mode::write>();
    Out1[0] = InitVal1;
    auto Out2 = OutBuf2.template get_access<access::mode::write>();
    Out2[0] = InitVal2;
    auto Out3 = OutBuf3.template get_access<access::mode::write>();
    Out3[0] = InitVal3;

    if (AllocType4 == usm::alloc::device) {
      Q.submit([&](handler &CGH) {
         CGH.single_task<KNameGroup<Name, class KernelNameUSM4>>(
             [=]() { *Out4 = InitVal4; });
       }).wait();
    } else {
      *Out4 = InitVal4;
    }
  }

  auto NDR = nd_range<1>{range<1>(NWorkItems), range<1>{WGSize}};
  if constexpr (IsSYCL2020) {
    Q.submit([&](handler &CGH) {
       auto In1 = InBuf1.template get_access<access::mode::read>(CGH);
       auto In2 = InBuf2.template get_access<access::mode::read>(CGH);
       auto In3 = InBuf3.template get_access<access::mode::read>(CGH);
       auto In4 = InBuf4.template get_access<access::mode::read>(CGH);

       auto Redu1 = sycl::reduction(OutBuf1, CGH, IdentityVal1, BOp1,
                                    getPropertyList<Mode1>());
       auto Redu2 = sycl::reduction(OutBuf2, CGH, IdentityVal2, BOp2,
                                    getPropertyList<Mode2>());
       auto Redu3 = sycl::reduction(OutBuf3, CGH, IdentityVal3, BOp3,
                                    getPropertyList<Mode3>());
       auto Redu4 =
           sycl::reduction(Out4, IdentityVal4, BOp4, getPropertyList<Mode4>());

       auto Lambda = [=](nd_item<1> NDIt, auto &Sum1, auto &Sum2, auto &Sum3,
                         auto &Sum4) {
         size_t I = NDIt.get_global_id(0);
         Sum1.combine(In1[I]);
         Sum2.combine(In2[I]);
         Sum3.combine(In3[I]);
         Sum4.combine(In4[I]);
       };
       CGH.parallel_for<Name>(NDR, Redu1, Redu2, Redu3, Redu4, Lambda);
     }).wait();
  } else {
    // Test ONEAPI reductions
    Q.submit([&](handler &CGH) {
       auto In1 = InBuf1.template get_access<access::mode::read>(CGH);
       auto In2 = InBuf2.template get_access<access::mode::read>(CGH);
       auto In3 = InBuf3.template get_access<access::mode::read>(CGH);
       auto In4 = InBuf4.template get_access<access::mode::read>(CGH);

       auto Out1 = OutBuf1.template get_access<Mode1>(CGH);
       auto Out2 = OutBuf2.template get_access<Mode2>(CGH);
       accessor<T3, 0, Mode3, access::target::global_buffer> Out3(OutBuf3, CGH);

       auto Redu1 = ONEAPI::reduction(Out1, IdentityVal1, BOp1);
       auto Redu2 = ONEAPI::reduction(Out2, IdentityVal2, BOp2);
       auto Redu3 = ONEAPI::reduction(Out3, IdentityVal3, BOp3);
       auto Redu4 = ONEAPI::reduction(Out4, IdentityVal4, BOp4);

       auto Lambda = [=](nd_item<1> NDIt, auto &Sum1, auto &Sum2, auto &Sum3,
                         auto &Sum4) {
         size_t I = NDIt.get_global_id(0);
         Sum1.combine(In1[I]);
         Sum2.combine(In2[I]);
         Sum3.combine(In3[I]);
         Sum4.combine(In4[I]);
       };
       CGH.parallel_for<Name>(NDR, Redu1, Redu2, Redu3, Redu4, Lambda);
     }).wait();
  }

  // Check the results and free memory.
  int Error = 0;
  {
    auto Out1 = OutBuf1.template get_access<access::mode::read>();
    auto Out2 = OutBuf2.template get_access<access::mode::read>();
    auto Out3 = OutBuf3.template get_access<access::mode::read>();

    T4 Out4Val;
    if (AllocType4 == usm::alloc::device) {
      buffer<T4, 1> Buf(&Out4Val, range<1>(1));
      Q.submit([&](handler &CGH) {
        auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
        CGH.copy(Out4, OutAcc);
      });
      Out4Val = (Buf.template get_access<access::mode::read>())[0];
    } else {
      Out4Val = *Out4;
    }

    Error += cherkResultIsExpected(1, CorrectOut1, Out1[0], IsSYCL2020) ? 0 : 1;
    Error += cherkResultIsExpected(2, CorrectOut2, Out2[0], IsSYCL2020) ? 0 : 1;
    Error += cherkResultIsExpected(3, CorrectOut3, Out3[0], IsSYCL2020) ? 0 : 1;
    Error += cherkResultIsExpected(4, CorrectOut4, Out4Val, IsSYCL2020) ? 0 : 1;
    free(Out4, Q.get_context());
  }

  if (Error)
    std::cerr << "The test failed for nd_range(" << NWorkItems << "," << WGSize
              << ")\n\n";

  return Error;
}

// Tests both implementations of reduction:
// sycl::reduction and sycl::ONEAPI::reduction
template <class Name, typename T1, access::mode Mode1, typename T2,
          access::mode Mode2, typename T3, access::mode Mode3, typename T4,
          access::mode Mode4, class BinaryOperation1, class BinaryOperation2,
          class BinaryOperation3, class BinaryOperation4>
int testBoth(queue &Q, T1 IdentityVal1, T1 InitVal1, BinaryOperation1 BOp1,
             T2 IdentityVal2, T2 InitVal2, BinaryOperation2 BOp2,
             T3 IdentityVal3, T3 InitVal3, BinaryOperation3 BOp3,
             T4 IdentityVal4, T3 InitVal4, BinaryOperation4 BOp4,
             usm::alloc AllocType4, size_t NWorkItems, size_t WGSize) {
  int Error =
      testOne<KName<Name, false>, false, T1, Mode1, T2, Mode2, T3, Mode3, T4,
              Mode4>(Q, IdentityVal1, InitVal1, BOp1, IdentityVal2, InitVal2,
                     BOp2, IdentityVal3, InitVal3, BOp3, IdentityVal4, InitVal4,
                     BOp4, AllocType4, NWorkItems, WGSize);

  Error +=
      testOne<KName<Name, true>, true, T1, Mode1, T2, Mode2, T3, Mode3, T4,
              Mode4>(Q, IdentityVal1, InitVal1, BOp1, IdentityVal2, InitVal2,
                     BOp2, IdentityVal3, InitVal3, BOp3, IdentityVal4, InitVal4,
                     BOp4, AllocType4, NWorkItems, WGSize);
  return Error;
}

int main() {
  queue Q;
  int Error = testBoth<class Case1, float, DW, int, RW, short, RW, int, RW>(
      Q, 0, 1000, std::plus<float>{}, 0, 2000, std::plus<>{}, 0, 4000,
      std::bit_or<>{}, 0, 8000, std::bit_xor<>{}, usm::alloc::shared, 16, 16);

  auto Add = [](auto x, auto y) { return (x + y); };
  Error += testBoth<class Case2, float, RW, int, RW, short, DW, int, DW>(
      Q, 0, 1000, std::plus<float>{}, 0, 2000, std::plus<>{}, 0, 4000, Add, 0,
      8000, std::plus<>{}, usm::alloc::device, 5 * (256 + 1), 5);

  if (!Error)
    std::cout << "Test passed\n";
  else
    std::cerr << Error << " test-cases failed\n";
  return Error;
}
