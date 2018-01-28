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
__global__ void SepCppBackwardKernel_fast(const int count,
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
  // intermediate storage for vertical gradients
  // no VRAM impact as it is located in the low-level cache
  Dtype cache[SEPCPP_FAST_KERNEL_LIMIT];
  for (int i = 0; i < kernel; ++i) cache[i] = Dtype(0);
  CUDA_KERNEL_LOOP(index, count) {
    // find the current location given the index
    const int x = index % width;
    const int y = (index / width) % height;
    const int pixels_in_channel = height * width;
    const int n = (index / pixels_in_channel) / channels;
    // both the horizontal and vertical gradients are accumulated within a
    // single pass; we iterate over each horizontal kernel element, calculating
    // its gradient and accumulating its contribution to each vertical kernel
    for (int j_ker = max(padding-x, 0), j_img = x - padding + j_ker;
        j_ker < kernel && j_img < width; ++j_ker, ++j_img) {
      Dtype v = Dtype(0);
      // retrieve the horizontal kernel element for vertical gradient calc.
      Dtype h_j = ker[(n*2*kernel + j_ker)*pixels_in_channel + y*width + x];
      for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
          i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
        Dtype t = Dtype(0);
        for (int c = 0; c < channels; ++c) {
          int offset = (n*channels + c)*pixels_in_channel;
          t += src[offset + y*width + x] *
               img[offset + i_img*width + j_img];
        }
        // v accumulates gradient for this horizontal kernel element
        v += t * ker[(n*2*kernel + kernel + i_ker)*pixels_in_channel +
                      y*width + x];
        // cache accumulates gradients for each vertical element, so we don't
        // have to run the entire second loop including channel accumulation
        cache[i_ker] += t * h_j;
      }
      diff[(n*2*kernel + j_ker)*pixels_in_channel + y*width + x] = v;
    }
    // vertical gradient is already calculated and stored in the cache, so all
    // there's left to do is copy it into the blob
    for (int i_ker = max(padding-y, 0), i_img = y - padding + i_ker;
        i_ker < kernel && i_img < height; ++i_ker, ++i_img) {
      diff[(n*2*kernel + kernel + i_ker)*pixels_in_channel +
            y*width + x] = cache[i_ker];
    }
  }
}

template <typename Dtype>
__global__ void SepCppBackwardKernel_any(const int count,
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
  // try to launch the optimized kernel, fall back to the default one
  if (kernel_ <= SEPCPP_FAST_KERNEL_LIMIT) {
    SepCppBackwardKernel_fast<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, batch_num_, channels_, height_, width_, kernel_, padding_,
        src_data, img_data, ker_data, diff_data);
  } else {
    SepCppBackwardKernel_any<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, batch_num_, channels_, height_, width_, kernel_, padding_,
        src_data, img_data, ker_data, diff_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SepCPPLayer);

}  // namespace caffe
