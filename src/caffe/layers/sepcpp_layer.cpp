#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/sepcpp_layer.hpp"

namespace caffe {

template <typename Dtype>
void SepCPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  kernel_ = 0;
  padding_ = 0;
  height_ = 0;
  width_ = 0;
  batch_num_ = 0;
  channels_ = 0;
}

template <typename Dtype>
void SepCPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  CHECK_EQ(B_input % 2, 0)
      << "kernel blob channel dimension must be an even integer.";
  int kernel = B_input / 2;
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
void SepCPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
          /* The same iterating technique as with the standard CPP layer
           * (double iterators per each loop for simultaneous boundary
           * checking within the source image and kernel).
           * Two loops over the neighborhood of the processed pixel:
           * the inner loop goes over each row, multiplying the pixels values
           * by their respective weights of the horizontal kernel (first half
           * of the kernel slice) and accumulating them
           * the outer loop collects those sums and multiplies them by
           * corresponding weights of the vertical kernel (second half).
           * Results are then stored in the output blob.
           */
          Dtype v = Dtype(0);
          for (int i_ker = std::max(padding_-y, 0),
              i_img = y - padding_ + i_ker;
              i_ker < kernel_ && i_img < height_; ++i_ker, ++i_img) {
            Dtype temp = Dtype(0);
            for (int j_ker = std::max(padding_-x, 0),
                j_img = x - padding_ + j_ker;
                j_ker < kernel_ && j_img < width_; ++j_ker, ++j_img) {
              temp += img[img_offset + i_img*width_ + j_img] *
                      ker[n*pixels_in_kblob +
                          j_ker * pixels_in_kernel +
                          pix_offset];
            }
            v += temp * ker[n*pixels_in_kblob +
                            (kernel_ + i_ker) * pixels_in_kernel +
                            pix_offset];
          }
          out[img_offset + pix_offset] = v;
        }
      }
    }
  }
}

template <typename Dtype>
void SepCPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* src = top[0]->cpu_diff();
  const Dtype* img = bottom[0]->cpu_data();
  const Dtype* ker = bottom[1]->cpu_data();
  Dtype* diff = bottom[1]->mutable_cpu_diff();

  int pixels_in_image = bottom[0]->count(1);
  int pixels_in_channel = bottom[0]->count(2);
  int pixels_in_kblob = bottom[1]->count(1);
  int pixels_in_kernel = bottom[0]->count(2);
  for (int n = 0; n < batch_num_; ++n) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        // Separate passes for horizontal and vertical kernels.
        // For horizontal pass, we iterate over each horizontal filter element,
        // for each of them iterating over the image and vertical filter
        // elements. Within the innermost loop we also consider channels.
        for (int j_ker = std::max(padding_-x, 0),
            j_img = x - padding_ + j_ker;
            j_ker < kernel_ && j_img < width_; ++j_ker, ++j_img) {
          Dtype v = Dtype(0);
          for (int i_ker = std::max(padding_-y, 0),
              i_img = y - padding_ + i_ker;
              i_ker < kernel_ && i_img < height_; ++i_ker, ++i_img) {
            // less RAM-efficient version
            Dtype t = Dtype(0);
            for (int c = 0; c < channels_; ++c) {
              int offset = n*pixels_in_image + c*pixels_in_channel;
              t += src[offset + y*width_ + x] *
                   img[offset + i_img*width_ + j_img];
            }
            v += t *
                 ker[n*pixels_in_kblob +
                    (kernel_ + i_ker) * pixels_in_kernel +
                    y*width_ + x];
          }
          diff[n*pixels_in_kblob +
               j_ker * pixels_in_kernel +
               y*width_ + x] = v;
        }
        // Vertical:
        for (int i_ker = std::max(padding_-y, 0),
            i_img = y - padding_ + i_ker;
            i_ker < kernel_ && i_img < height_; ++i_ker, ++i_img) {
          Dtype v = Dtype(0);
          for (int j_ker = std::max(padding_-x, 0),
              j_img = x - padding_ + j_ker;
              j_ker < kernel_ && j_img < width_; ++j_ker, ++j_img) {
            Dtype t = Dtype(0);
            for (int c = 0; c < channels_; ++c) {
              int offset = n*pixels_in_image + c*pixels_in_channel;
              t += src[offset + y*width_ + x] *
                   img[offset + i_img*width_ + j_img];
            }
            v += t *
                 ker[n*pixels_in_kblob +
                    j_ker * pixels_in_kernel +
                    y*width_ + x];
          }
          diff[n*pixels_in_kblob +
               (kernel_ + i_ker) * pixels_in_kernel +
               y*width_ + x] = v;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SepCPPLayer);
#endif

INSTANTIATE_CLASS(SepCPPLayer);
REGISTER_LAYER_CLASS(SepCPP);

}  // namespace caffe
