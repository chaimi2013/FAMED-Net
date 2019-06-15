#include <algorithm>
#include <vector>

#include "caffe/layers/brelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void BReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype minval = this -> layer_param_.brelu_param().minval();
  Dtype maxval = this -> layer_param_.brelu_param().maxval();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min(std::max(bottom_data[i], minval), maxval);
  }
}

template <typename Dtype>
void BReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype minval = this -> layer_param_.brelu_param().minval();
    Dtype maxval = this -> layer_param_.brelu_param().maxval();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] >= minval)&&(bottom_data[i] <= maxval));
    }
  }
}

INSTANTIATE_CLASS(BReLULayer);
REGISTER_LAYER_CLASS(BReLU);

}  // namespace caffe
