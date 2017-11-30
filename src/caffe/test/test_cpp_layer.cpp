#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/cpp_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void reference_conv(Blob<Dtype>* image_, Blob<Dtype>* kernels_,
    Blob<Dtype>* output_) {
  /* Calculate each pixel of the output separately, propagating an entire image
   * through a standard Convolution Layer, but each time set a different filter
   * (specific for that particular pixel).
   */
  // Extract parameters
  // image_ and kernels_ are assumed to be checked by the CPP Layer and thus
  // correct - no further checks are made here
  // output_ however will be reshaped by Convolution Layer
  int kernel = static_cast<int>(std::sqrt(kernels_->shape(1)));
  int padding = (kernel - 1) / 2;
  int num_batch = image_->shape(0);
  int channel = image_->shape(1);
  int height = image_->shape(2);
  int width = image_->shape(3);
  // Set up the reference layer
  LayerParameter layer_param;
  ConvolutionParameter* reference_layer_param =
      layer_param.mutable_convolution_param();
  reference_layer_param->add_kernel_size(kernel);
  reference_layer_param->add_stride(1);
  reference_layer_param->add_pad(padding);
  reference_layer_param->set_group(channel);
  reference_layer_param->set_num_output(channel);
  reference_layer_param->mutable_bias_filler()->set_type("constant");
  reference_layer_param->mutable_bias_filler()->set_value(0.0);
  ConvolutionLayer<Dtype> reference_layer(layer_param);
  // Each iteration, Convolution writes to "intermediate_output"; then
  // we extract the results pixel to pixel and copy to the "output" blob.
  Blob<Dtype> intermediate_output;
  vector<Blob<Dtype>*> inputs, outputs;
  inputs.push_back(image_);
  outputs.push_back(&intermediate_output);
  reference_layer.SetUp(inputs, outputs);
  // Forward once to allocate blobs
  reference_layer.Forward(inputs, outputs);
  output_->ReshapeLike(intermediate_output);
  // Convenience handle to the conv filter blob
  Blob<Dtype>* reference_filter = reference_layer.blobs()[0].get();
  for (int n = 0; n < num_batch; ++n) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // For each pixel, copy the corresponding filter to the convolution
        // filter bank...
        for (int i = 0; i < kernel; ++i) {
          for (int j = 0; j < kernel; ++j) {
            for (int k = 0; k < channel; ++k) {
              reference_filter->mutable_cpu_data()[
                  reference_filter->offset(k, 0, i, j)] =
                  kernels_->data_at(n, i*kernel+j, y, x);
            }
          }
        }
        // ...forward the whole layer...
        reference_layer.Forward(inputs, outputs);
        // ...and copy the output to the output blob
        for (int k = 0; k < channel; ++k) {
          int offset = output_->offset(n, k, y, x);
          output_->mutable_cpu_data()[offset] =
              intermediate_output.cpu_data()[offset];
        }
      }
    }
  }
}

template <typename Dtype>
class CPPLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  CPPLayerTest()
      : blob_bottom_img_(new Blob<Dtype>(2, 3, 12, 16)),
        blob_bottom_ker_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_img_);
    blob_bottom_vec_.push_back(blob_bottom_img_);
    blob_bottom_vec_.push_back(blob_bottom_ker_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void PrepareKernelBlob(int kernel_size) {
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    vector<int> shape;
    shape.push_back(this->blob_bottom_img_->shape(0));
    shape.push_back(kernel_size * kernel_size);
    shape.push_back(this->blob_bottom_img_->shape(2));
    shape.push_back(this->blob_bottom_img_->shape(3));
    this->blob_bottom_ker_->Reshape(shape);
    filler.Fill(this->blob_bottom_ker_);
  }

  virtual ~CPPLayerTest() {
    delete blob_bottom_img_;
    delete blob_bottom_ker_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_img_;
  Blob<Dtype>* const blob_bottom_ker_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CPPLayerTest, TestDtypes);

TYPED_TEST(CPPLayerTest, TestForward3) {
  /* Test forward pass against the existing convolution implementation. */
  int kernel = 3;
  // Create the filter blob for the given kernel size
  this->PrepareKernelBlob(kernel);
  // Propagate using CPP Layer
  LayerParameter layer_param;
  CPPLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Calculate the reference implementation
  Blob<TypeParam> reference_output;
  reference_conv<TypeParam>(this->blob_bottom_img_, this->blob_bottom_ker_,
      &reference_output);
  // Compare
  CHECK_EQ(this->blob_top_->count(), reference_output.count()) <<
      "CPP output and reference convolution output are not equal size!";
  for (int i = 0; i < reference_output.count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
        reference_output.cpu_data()[i], 1e-3);
  }
}

TYPED_TEST(CPPLayerTest, TestForward5) {
  /* Test forward pass against the existing convolution implementation. */
  int kernel = 5;
  // Create the filter blob for the given kernel size
  this->PrepareKernelBlob(kernel);
  // Propagate using CPP Layer
  LayerParameter layer_param;
  CPPLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Calculate the reference implementation
  Blob<TypeParam> reference_output;
  reference_conv<TypeParam>(this->blob_bottom_img_, this->blob_bottom_ker_,
      &reference_output);
  // Compare
  CHECK_EQ(this->blob_top_->count(), reference_output.count()) <<
      "CPP output and reference convolution output are not equal size!";
  for (int i = 0; i < reference_output.count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
        reference_output.cpu_data()[i], 1e-3);
  }
}

TYPED_TEST(CPPLayerTest, TestForward7) {
  /* Test forward pass against the existing convolution implementation. */
  int kernel = 7;
  // Create the filter blob for the given kernel size
  this->PrepareKernelBlob(kernel);
  // Propagate using CPP Layer
  LayerParameter layer_param;
  CPPLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Calculate the reference implementation
  Blob<TypeParam> reference_output;
  reference_conv<TypeParam>(this->blob_bottom_img_, this->blob_bottom_ker_,
      &reference_output);
  // Compare
  CHECK_EQ(this->blob_top_->count(), reference_output.count()) <<
      "CPP output and reference convolution output are not equal size!";
  for (int i = 0; i < reference_output.count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i],
        reference_output.cpu_data()[i], 1e-3);
  }
}

TYPED_TEST(CPPLayerTest, TestGradient3) {
  int kernel = 3;
  // Create the filter blob for the given kernel size
  this->PrepareKernelBlob(kernel);
  // Create the layer and GradientChecker
  LayerParameter layer_param;
  CPPLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Verify gradient for each input pixel
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    checker.CheckGradientSingle(&layer,
        this->blob_bottom_vec_, this->blob_top_vec_, 1, 0, i);
  }
}
}  // namespace caffe
