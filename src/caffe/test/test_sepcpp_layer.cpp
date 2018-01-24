#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/cpp_layer.hpp"
#include "caffe/layers/sepcpp_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void reference_cpp(Blob<Dtype>* image_, Blob<Dtype>* kernels_,
    Blob<Dtype>* output_) {
  /* Calculate the reference result via the CPP layer, by expanding the
   * separable filters into standard CPP kernel blob.
   */
  // Prepare the blob for expanded filters
  int kernel = kernels_->shape(1) / 2;
  vector<int> ref_shape;
  ref_shape.push_back(kernels_->shape(0));
  ref_shape.push_back(kernel * kernel);
  ref_shape.push_back(kernels_->shape(2));
  ref_shape.push_back(kernels_->shape(3));
  Blob<Dtype> expanded_kernels;
  expanded_kernels.Reshape(ref_shape);
  // Expand the filters
  for (int n = 0; n < kernels_->shape(0); ++n) {
    for (int y = 0; y < kernels_->shape(2); ++y) {
      for (int x = 0; x < kernels_->shape(3); ++x) {
        for (int j = 0; j < kernel; ++j) {
          for (int i = 0; i < kernel; ++i) {
            expanded_kernels.mutable_cpu_data()[
                expanded_kernels.offset(n, j*kernel + i, y, x)] =
                kernels_->data_at(n, j + kernel, y, x) *
                kernels_->data_at(n, i, y, x);
          }
        }
      }
    }
  }
  // Set up the reference layer
  vector<Blob<Dtype>*> inputs, outputs;
  inputs.push_back(image_);
  inputs.push_back(&expanded_kernels);
  outputs.push_back(output_);
  LayerParameter layer_param;
  CPPLayer<Dtype> reference_layer(layer_param);
  reference_layer.SetUp(inputs, outputs);
  // Forward to obtain the results
  reference_layer.Forward(inputs, outputs);
}

template <typename TypeParam>
class SepCPPLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SepCPPLayerTest()
      : blob_bottom_img_(new Blob<Dtype>(2, 3, 5, 8)),
        blob_bottom_ker_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    FillerParameter filler_param;
    filler_param.set_min(0.);
    filler_param.set_max(1.);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_img_);
    blob_bottom_vec_.push_back(blob_bottom_img_);
    blob_bottom_vec_.push_back(blob_bottom_ker_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void PrepareKernelBlob(int kernel_size) {
    FillerParameter filler_param;
    filler_param.set_min(0.);
    filler_param.set_max(1.);
    UniformFiller<Dtype> filler(filler_param);
    vector<int> shape;
    shape.push_back(this->blob_bottom_img_->shape(0));
    shape.push_back(2 * kernel_size);
    shape.push_back(this->blob_bottom_img_->shape(2));
    shape.push_back(this->blob_bottom_img_->shape(3));
    this->blob_bottom_ker_->Reshape(shape);
    filler.Fill(this->blob_bottom_ker_);
  }

  virtual void ForwardTest(int kernel_size) {
    /* Test forward pass against the existing CPP implementation. */
    typedef typename TypeParam::Dtype Dtype;
    // Create the filter blob for the given kernel size
    this->PrepareKernelBlob(kernel_size);
    // Propagate using the SCPP layer
    LayerParameter layer_param;
    SepCPPLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Calculate the reference result using CPP
    Blob<Dtype> reference_output;
    reference_cpp(this->blob_bottom_img_, this->blob_bottom_ker_,
        &reference_output);
    // Compare
    CHECK_EQ(this->blob_top_->count(), reference_output.count()) <<
        "SCPP output and the reference CPP output are not equal size!";
    for (int i = 0; i < reference_output.count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i],
          reference_output.cpu_data()[i], 1e-3);
    }
  }

  virtual void BackwardTest(int kernel_size) {
    /* Test backward pass using Caffe's built-in gradient checker class. */
    typedef typename TypeParam::Dtype Dtype;
    // Create the filter blob for the given kernel size
    this->PrepareKernelBlob(kernel_size);
    // Set up the layer
    LayerParameter layer_param;
    SepCPPLayer<Dtype> layer(layer_param);
    // Let the GradientChecker do the rest
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
      checker.CheckGradientSingle(&layer,
          this->blob_bottom_vec_, this->blob_top_vec_, 1, 0, i);
    }
  }

  virtual ~SepCPPLayerTest() {
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

TYPED_TEST_CASE(SepCPPLayerTest, TestDtypesAndDevices);

TYPED_TEST(SepCPPLayerTest, TestForward3) {
  this->ForwardTest(3);
}

TYPED_TEST(SepCPPLayerTest, TestForward5) {
  this->ForwardTest(5);
}

TYPED_TEST(SepCPPLayerTest, TestForward7) {
  this->ForwardTest(7);
}

TYPED_TEST(SepCPPLayerTest, TestForward25) {
  this->ForwardTest(25);
}

TYPED_TEST(SepCPPLayerTest, TestGradient3) {
  this->BackwardTest(3);
}

TYPED_TEST(SepCPPLayerTest, TestGradient5) {
  this->BackwardTest(5);
}

TYPED_TEST(SepCPPLayerTest, TestGradient11) {
  this->BackwardTest(11);
}

// This needs increasing the image size beyond 25 pixels, which makes the
// computation VERY heavy. Run at your own risk.
/*TYPED_TEST(SepCPPLayerTest, ExtremeTest) {
  this->ForwardTest(25);
  this->BackwardTest(25);
}*/

}  // namespace caffe
