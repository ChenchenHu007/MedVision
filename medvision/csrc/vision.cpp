#include "roi_align_3d.h"
#include "roi_align_rotated_3d.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // operators
    
    m.def("roi_align_3d_forward", &roi_align_3d_forward, "roi_align_3d_forward");
    m.def("roi_align_3d_backward", &roi_align_3d_backward, "roi_align_3d_backward");

    m.def("roi_align_rotated_3d_forward", &roi_align_rotated_3d_forward,
    "roi_align_rotated_3d_forward");
    m.def("roi_align_rotated_3d_backward", &roi_align_rotated_3d_backward,
    "roi_align_rotated_3d_backward");

}