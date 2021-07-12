// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop_task.

// Level-Zero
#include <level_zero/ze_api.h>

// SYCL
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

using namespace sycl;

constexpr size_t SIZE = 16;

int main() {
  queue queue{};

  try {
    buffer<uint8_t, 1> buffer(SIZE);
    image<2> image(image_channel_order::rgba, image_channel_type::fp32,
                   {SIZE, SIZE});

    ze_context_handle_t ze_context =
        queue.get_context().get_native<backend::level_zero>();

    queue
        .submit([&](handler &cgh) {
          auto buffer_acc = buffer.get_access<access::mode::write>(cgh);
          auto image_acc = image.get_access<float4, access::mode::write>(cgh);
          cgh.interop_task([=](const interop_handler &ih) {
            void *device_ptr = ih.get_mem<backend::level_zero>(buffer_acc);
            ze_memory_allocation_properties_t memAllocProperties{};
            ze_result_t res = zeMemGetAllocProperties(
                ze_context, device_ptr, &memAllocProperties, nullptr);
            assert(res == ZE_RESULT_SUCCESS);

            ze_image_handle_t ze_image =
                ih.get_mem<backend::level_zero>(image_acc);
            assert(ze_image != nullptr);
          });
        })
        .wait();
  } catch (exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  } catch (const char *msg) {
    std::cout << "Exception caught: " << msg << std::endl;
    return 1;
  }

  return 0;
}
