// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

// XFAIL: cuda

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)


/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    REPEAT address_mode and LINEAR filter_mode

*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

using pixelT = sycl::float4;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::float4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

// some constants.

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto repeat = addressing_mode::repeat;
constexpr auto linear = filtering_mode::linear;

void test_normalized_repeat_linear_sampler(image_channel_order ChanOrder,
                                           image_channel_type ChanType) {
  int numTests = 11; // drives the size of the testResults buffer, and the
                     // number of report iterations. Kludge.

  // we'll use these four pixels for our image. Makes it easy to measure
  // interpolation and spot "off-by-one" probs.
  // These values will work consistently with different levels of float
  // precision (like unorm_int8 vs. fp32)
  pixelT leftEdge{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT body{0.6f, 0.4f, 0.2f, 0.0f};
  pixelT bony{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT rightEdge{0.6f, 0.4f, 0.2f, 0.0f};

  queue Q;
  const sycl::range<1> ImgRange_1D(width);
  { // closure
    // - create an image
    image<1> image_1D(ChanOrder, ChanType, ImgRange_1D);
    event E_Setup = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::write>(cgh);
      cgh.single_task<class setupUnormLinear>([=]() {
        image_acc.write(0, leftEdge);
        image_acc.write(1, body);
        image_acc.write(2, bony);
        image_acc.write(3, rightEdge);
      });
    });
    E_Setup.wait();

    // use a buffer to report back test results.
    buffer<pixelT, 1> testResults((range<1>(numTests)));

    // sampler
    auto Norm_Repeat_Linear_sampler = sampler(normalized, repeat, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_norm_linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // clang-format off
        // Normalized Pixel Locations.  
        //      .125        .375        .625        .875            <-- exact center
        //  |-----^-----|-----^-----|-----^-----|-----^-----
        //[0.0         .25         .50         .75          (1)     <-- low boundary (included in pixel)
        //                                                              upper boundary inexact. (e.g. .2499999)
        // clang-format on

        // 0-2 read three pixels at inner boundary locations,  sample:
        // Normalized +  Repeat  + Linear
        test_acc[i++] = image_acc.read(
            0.25f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.50f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.75f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}

        // 3-6 read four pixels above right bound,   sample: Normalized + Repeat
        // + Linear
        test_acc[i++] = image_acc.read(
            1.0f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.25f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.5f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.75f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        // 7-10 read four pixels below left bound. sample: Normalized + Repeat +
        // Linear
        test_acc[i++] = image_acc.read(
            -0.75f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            -0.5f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            -0.25f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.0f, Norm_Repeat_Linear_sampler); // {0.4,0.4,0.4,0.4}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    auto test_acc = testResults.get_access<access::mode::read>();
    for (int i = 0, idx = 0; i < numTests; i++, idx++) {
      if (i == 0) {
        idx = 1;
        std::cout << "read three pixels at inner boundary locations,  sample:  "
                     " Normalized +  Repeat  + Linear"
                  << std::endl;
      }
      if (i == 3) {
        idx = 0;
        std::cout << "read four pixels above right bound,   sample: Normalized "
                     "+ Repeat + Linear"
                  << std::endl;
      }
      if (i == 7) {
        idx = 0;
        std::cout << "read four pixels below left bound. sample: Normalized + "
                     "Repeat + Linear"
                  << std::endl;
      }
      pixelT testPixel = test_acc[i];
      std::cout << i << " -- " << idx << ": ";
      outputPixel(testPixel);
      std::cout << std::endl;
    }
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  if (D.has(aspect::image)) {
    // the _int8 channels are one byte per channel, or four bytes per pixel (for
    // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
    // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
    // channel, or sixteen bytes per pixel (for RGBA).
    // CUDA has limited support for image_channel_type, so the tests use

    std::cout << "fp32 -------------" << std::endl;
    test_normalized_repeat_linear_sampler(image_channel_order::rgba,
                                          image_channel_type::fp32);

    std::cout << "unorm_int8 -------" << std::endl;
    test_normalized_repeat_linear_sampler(image_channel_order::rgba,
                                          image_channel_type::unorm_int8);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: fp32 -------------
// CHECK-NEXT: read three pixels at inner boundary locations,  sample:   Normalized +  Repeat  + Linear
// CHECK-NEXT: 0 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 1 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 2 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: read four pixels above right bound,   sample: Normalized + Repeat + Linear
// CHECK-NEXT: 3 -- 0: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 4 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 5 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 6 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: read four pixels below left bound. sample: Normalized + Repeat + Linear
// CHECK-NEXT: 7 -- 0: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 8 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 9 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 10 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: unorm_int8 -------
// CHECK-NEXT: read three pixels at inner boundary locations,  sample:   Normalized +  Repeat  + Linear
// CHECK-NEXT: 0 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 1 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 2 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: read four pixels above right bound,   sample: Normalized + Repeat + Linear
// CHECK-NEXT: 3 -- 0: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 4 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 5 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 6 -- 3: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: read four pixels below left bound. sample: Normalized + Repeat + Linear
// CHECK-NEXT: 7 -- 0: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 8 -- 1: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 9 -- 2: {0.4,0.4,0.4,0.4} 
// CHECK-NEXT: 10 -- 3: {0.4,0.4,0.4,0.4}
// clang-format on