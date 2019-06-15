#include <algorithm>
#include <vector>

#include "caffe/layers/brelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype minval, Dtype maxval) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] < 1 ? (in[index] > 0 ? in[index] : in[index] * minval) : maxval;
  }
}

template <typename Dtype>
void BReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype minval = this -> layer_param_.brelu_param().minval();
  Dtype maxval = this -> layer_param_.brelu_param().maxval();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, minval, maxval);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void BReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype minval, Dtype maxval) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0 && in_data[index] <= 1)
        + (in_data[index] <= 0 || in_data[index] >= maxval) * minval);
  }
}

template <typename Dtype>
void BReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();  
    Dtype minval = this -> layer_param_.brelu_param().minval();
    Dtype maxval = this -> layer_param_.brelu_param().maxval();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, minval, maxval);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(BReLULayer);


}  // namespace caffe
