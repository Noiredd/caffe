#include <algorithm>
#include <vector>

#include "caffe/layers/cpp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CppForwardKernel(const int count,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernel,
    const int padding,
    const Dtype* img,
    const Dtype* ker,
    Dtype* top) {
  CUDA_KERNEL_LOOP(index, count) {
    // find the current location given the index
    const int x = index % width;
    const int y = (index / width) % height;
    const int pixels_in_channel = height * width;
    const int c = (index / pixels_in_channel) % channels;
    const int n = (index / pixels_in_channel) / channels;
    // pretty much the same thing as with the CPU
    Dtype v = Dtype(0);
    for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
        i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
      for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
          j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
        v += img[(n*channels + c)*pixels_in_channel + i_img*width + j_img] *
             ker[(n*kernel*kernel + i_ker*kernel + j_ker)*pixels_in_channel +
                 y*width + x];
      }
    }
    top[index] = v;
  }
}

template <typename Dtype>
void CPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* img_data = bottom[0]->gpu_data();
  const Dtype* ker_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // launch a cuda kernel for each of the output pixels
  const int count = bottom[0]->count();
  CppForwardKernel<Dtype>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, batch_num_, channels_, height_, width_, kernel_, padding_,
      img_data, ker_data, top_data);
}

template <typename Dtype>
__global__ void CppBackwardKernel(const int count,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernel,
    const int padding,
    const Dtype* img,
    const Dtype* src,
    Dtype* diff) {
  CUDA_KERNEL_LOOP(index, count) {
    // find the current location given the index
    const int x = index % width;
    const int y = (index / width) % height;
    const int pixels_in_channel = height * width;
    const int n = (index / pixels_in_channel) / channels;
    // loop over each element of this filter
    for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
        i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
      for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
          j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
        Dtype v = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          int offset = (n*channels + c)*pixels_in_channel;
          v += src[offset + y*width + x] *
               img[offset + i_img*width + j_img];
        }
        diff[(n*kernel*kernel + i_ker*kernel + j_ker)*pixels_in_channel +
             y*width + x] = v;
      }
    }
  }
}

template <typename Dtype>
void CPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* src_data = top[0]->gpu_diff();
  Dtype* diff_data = bottom[1]->mutable_gpu_diff();
  const Dtype* img_data = bottom[0]->gpu_data();
  // launch a cuda kernel for each of the image pixels
  // will calculate gradients for each element of the corresponding filter
  const int count = bottom[0]->count();
  CppBackwardKernel<Dtype>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, batch_num_, channels_, height_, width_, kernel_, padding_,
      img_data, src_data, diff_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(CPPLayer);

}  // namespace caffe
