#ifndef CAFFE_SEPCPP_LAYER_HPP_
#define CAFFE_SEPCPP_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/math_functions.hpp"

// Cutoff limit for the optimized backward pass GPU kernel.
// Needs to be defined, as this is also the size of GPU cache for the gradient
// calculation (needs to be known at compile-time).
// No effect on CPU code.
#define SEPCPP_FAST_KERNEL_LIMIT 101

namespace caffe {

/**
 * @brief Performs a 2D grouped separable convolution on an Nx3xHxW blob
 * using a bank of filters given by an NxBxHxW (where B=K+K and K is an
 * odd number - kernel size), with a separate kernel for each image pixel.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class SepCPPLayer : public Layer<Dtype> {
 public:
  explicit SepCPPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SepCPP"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_;
  int padding_;
  int height_;
  int width_;
  int batch_num_;
  int channels_;
};

}  // namespace caffe

#endif  // CAFFE_SEPCPP_LAYER_HPP_
