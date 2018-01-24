#include <algorithm>
#include <vector>

#include "caffe/layers/sepcpp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SepCppForwardKernel(const int count,
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
    // iterate over the pixel and its surroundings
    Dtype v = Dtype(0), t;
    for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
        i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
      t = Dtype(0);
      for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
          j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
        t += img[(n*channels + c)*pixels_in_channel + i_img*width + j_img] *
             ker[(n*2*kernel + j_ker)*pixels_in_channel + y*width + x];
      }
      v += t * ker[(n*2*kernel + kernel + i_ker)*pixels_in_channel +
                    y*width + x];
    }
    top[index] = v;
  }
}

template <typename Dtype>
void SepCPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* img_data = bottom[0]->gpu_data();
  const Dtype* ker_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  SepCppForwardKernel<Dtype>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, batch_num_, channels_, height_, width_, kernel_, padding_,
      img_data, ker_data, top_data);
}

template <typename Dtype>
__global__ void SepCppBackwardKernel(const int count,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int kernel,
    const int padding,
    const Dtype* src,
    const Dtype* img,
    const Dtype* ker,
    Dtype* diff) {
  CUDA_KERNEL_LOOP(index, count) {
    // find the current location given the index
    const int x = index % width;
    const int y = (index / width) % height;
    const int pixels_in_channel = height * width;
    const int n = (index / pixels_in_channel) / channels;
    // horizontal gradient loop
    for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
        j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
      Dtype v = Dtype(0);
      for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
          i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
        Dtype t = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          int offset = (n*channels + c)*pixels_in_channel;
          t += src[offset + y*width + x] *
               img[offset + i_img*width + j_img];
        }
        v += t * ker[(n*2*kernel + kernel + i_ker)*pixels_in_channel +
                      y*width + x];
      }
      diff[(n*2*kernel + j_ker)*pixels_in_channel + y*width + x] = v;
    }
    // vertical gradient loop
    for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
        i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
      Dtype v = Dtype(0);
      for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
          j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
        Dtype t = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          int offset = (n*channels + c)*pixels_in_channel;
          t += src[offset + y*width + x] *
               img[offset + i_img*width + j_img];
        }
        v += t * ker[(n*2*kernel + j_ker)*pixels_in_channel + y*width + x];
      }
      diff[(n*2*kernel + kernel + i_ker)*pixels_in_channel +
            y*width + x] = v;
    }
  }
}

template <typename Dtype>
void SepCPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* src_data = top[0]->gpu_diff();
  const Dtype* img_data = bottom[0]->gpu_data();
  const Dtype* ker_data = bottom[1]->gpu_data();
  Dtype* diff_data = bottom[1]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  SepCppBackwardKernel<Dtype>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, batch_num_, channels_, height_, width_, kernel_, padding_,
      src_data, img_data, ker_data, diff_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(SepCPPLayer);

}  // namespace caffe
