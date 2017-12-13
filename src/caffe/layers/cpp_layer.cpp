#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/cpp_layer.hpp"

namespace caffe {

template <typename Dtype>
void CPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  kernel_ = 0;
  padding_ = 0;
  height_ = 0;
  width_ = 0;
  batch_num_ = 0;
  channels_ = 0;
}

template <typename Dtype>
void CPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // TODO: forbid backprop to image blob
  // Check input blob dimensions
  CHECK_EQ(bottom[0]->num_axes(), 4)
      << "inputs must have exactly 4 axes.";
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "both input blobs must have the same number of axes.";
  // Check batch dimension equality
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "both input blobs must have the same batch dimension length.";
  batch_num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  // Check kernel dimension correctness
  int B_input = bottom[1]->shape(1);
  int kernel = static_cast<int>((sqrt(static_cast<float>((B_input)))));
  CHECK_EQ(kernel*kernel, B_input)
      << "input channel dimension must be a square of an integer.";
  CHECK_EQ(kernel % 2, 1)
      << "input kernel size must be an odd integer.";
  kernel_ = kernel;
  padding_ = (kernel - 1) / 2;
  // Check input spatial dimensions
  CHECK_EQ(bottom[0]->shape(-1), bottom[1]->shape(-1))
      << "inputs must have the same height.";
  CHECK_EQ(bottom[0]->shape(-2), bottom[1]->shape(-2))
      << "inputs must have the same width.";
  width_ = bottom[0]->shape(-1);
  height_ = bottom[0]->shape(-2);
  // Shape the output blob
  top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void CPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* img = bottom[0]->cpu_data();
  const Dtype* ker = bottom[1]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();

  int pixels_in_image = bottom[0]->count(1);
  int pixels_in_channel = bottom[0]->count(2);
  int pixels_in_kblob = bottom[1]->count(1);
  int pixels_in_kernel = bottom[1]->count(2);
  for (int n = 0; n < batch_num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      int img_offset = n*pixels_in_image + c*pixels_in_channel;
      int pix_offset = 0;
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x, ++pix_offset) {
          /* Smart inner loops, simultaneously over kernel and source image.
           * Start by checking the starting point - if the position within the
           * output is smaller than padding, we cannot try to access pixels
           * outside the boundary. So we set the kernel iterator to some
           * non-negative value (std::max). Then we select the image iterator
           * (which is in absolute coordinates in the image space) accordingly.
           * Each iteration we increment both iterators - to move within both
           * the image an kernel spaces simultaneously. The loop ends when
           * either we are about to exceed the kernel space (it_ker < kernel_)
           * or we are about to exceed the image space (it_img < dim_ - 1).
           */
          Dtype v = Dtype(0);
          for (int i_ker = std::max(padding_-y, 0),
              i_img = y - padding_ + i_ker;
              i_ker < kernel_ && i_img < height_; ++i_ker, ++i_img) {
            for (int j_ker = std::max(padding_-x, 0),
                j_img = x - padding_ + j_ker;
                j_ker < kernel_ && j_img < width_; ++j_ker, ++j_img) {
              v +=  img[img_offset + i_img*width_ + j_img] *
                    ker[n*pixels_in_kblob +
                        (i_ker*kernel_+j_ker)*pixels_in_kernel +
                        pix_offset];
            }
          }
          out[img_offset + pix_offset] = v;
        }
      }
    }
  }
}

template <typename Dtype>
void CPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* src = top[0]->cpu_diff();
  Dtype* diff = bottom[1]->mutable_cpu_diff();
  const Dtype* img = bottom[0]->cpu_data();

  int pixels_in_image = bottom[0]->count(1);
  int pixels_in_channel = bottom[0]->count(2);
  int pixels_in_kblob = bottom[1]->count(1);
  int pixels_in_kernel = bottom[0]->count(2);
  for (int n = 0; n < batch_num_; ++n) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        // Smart loops like with the forward pass
        for (int i_ker = std::max(padding_-y, 0),
            i_img = y - padding_ + i_ker;
            i_ker < kernel_ && i_img < height_; ++i_ker, ++i_img) {
          for (int j_ker = std::max(padding_-x, 0),
              j_img = x - padding_ + j_ker;
              j_ker < kernel_ && j_img < width_; ++j_ker, ++j_img) {
            Dtype v = Dtype(0);
            for (int c = 0; c < channels_; ++c) {
              int offset = n*pixels_in_image + c*pixels_in_channel;
              v += src[offset + y*width_ + x] *
                   img[offset + i_img*width_ + j_img];
            }
            diff[n*pixels_in_kblob +
                 (i_ker*kernel_+j_ker)*pixels_in_kernel +
                 y*width_ + x] = v;
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CPPLayer);
#endif

INSTANTIATE_CLASS(CPPLayer);
REGISTER_LAYER_CLASS(CPP);

}  // namespace caffe
