#include "config.h"
#include "io/staged_reader.h"
#include "operators/argmax_op.h"
#include "operators/attention_op.h"
#include "operators/embedding_op.h"
#include "operators/ffn_op.h"
#include "operators/linear_op.h"
#include "operators/matmul_op.h"
#include "operators/qkv_projection_op.h"
#include "operators/residual_add_op.h"
#include "operators/rmsnorm_op.h"
#include "operators/rope_op.h"
#include "reference_fixture.h"
#include "runtime/cuda_context.h"
#include "runtime/device_buffer.h"
#include "runtime/model_weights_gpu.h"
#include "runtime/rope_tables.h"
#include "runtime/workspace.h"
#include "model/decoder_block.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace {

constexpr const char* kFixtureDir = "tests/data/reference";
constexpr float kSyntheticTolerance = 1e-5f;
constexpr float kModelTolerance = 1e-2f;
constexpr size_t kNumQHeads = 32;
constexpr size_t kNumKVHeads = 8;
constexpr size_t kHeadDim = 128;
constexpr size_t kMaxPositions = 1024;

runtime::DeviceTensorView<const float> const_view(runtime::DeviceTensorView<float> tensor) {
  return runtime::DeviceTensorView<const float>(tensor.data, tensor.shape);
}

bool has_prefix(const string& value, const string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

bool check_exact_i32(const string& label, const vector<int>& got, const vector<int>& expected) {
  if (got.size() != expected.size()) {
    cout << "  " << label << " size mismatch: got " << got.size() << " expected "
         << expected.size() << "\n";
    return false;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      cout << "  " << label << " mismatch at index " << i << " (got=" << got[i]
           << " expected=" << expected[i] << ")\n";
      return false;
    }
  }
  return true;
}

bool check_max_abs(const string& label, const vector<float>& got, const vector<float>& expected,
                   float tolerance) {
  if (got.size() != expected.size()) {
    cout << "  " << label << " size mismatch: got " << got.size() << " expected "
         << expected.size() << "\n";
    return false;
  }

  float max_err = 0.0f;
  size_t worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float err = std::fabs(got[i] - expected[i]);
    if (err > max_err) {
      max_err = err;
      worst = i;
    }
  }

  if (max_err > tolerance) {
    cout << "  " << label << " max |err|=" << max_err << " at index " << worst
         << " (got=" << got[worst] << " expected=" << expected[worst]
         << ") tolerance=" << tolerance << "\n";
    return false;
  }
  return true;
}

runtime::DeviceBuffer<float> upload_f32(const vector<float>& host, const runtime::CudaContext& context) {
  runtime::DeviceBuffer<float> buffer(host.size());
  if (!host.empty())
    buffer.copy_from_host(host.data(), host.size(), context.stream());
  return buffer;
}

vector<float> download_f32(const runtime::DeviceBuffer<float>& buffer, size_t count,
                           const runtime::CudaContext& context) {
  vector<float> host(count);
  if (!host.empty())
    buffer.copy_to_host(host.data(), host.size(), context.stream());
  context.synchronize();
  return host;
}

vector<int> download_i32(const runtime::DeviceBuffer<int>& buffer, size_t count,
                         const runtime::CudaContext& context) {
  vector<int> host(count);
  if (!host.empty())
    buffer.copy_to_host(host.data(), host.size(), context.stream());
  context.synchronize();
  return host;
}

struct ModelState {
  runtime::CudaContext context;
  io::StagedReader staged_reader;
  runtime::ModelWeightsGPU weights;
  runtime::RoPETables rope_tables;
  EmbeddingOp embedding;
  vector<DecoderBlock> blocks;
  RmsNormOp final_norm;
  LinearOp lm_head;
  ArgmaxOp argmax;
  runtime::Workspace workspace;
  size_t hidden_dim;
  size_t vocab_size;

  ModelState()
      : staged_reader(1 << 20),
        weights(WEIGHTS_DIR_PATH, staged_reader, context),
        rope_tables(context, kMaxPositions, kHeadDim, ROPE_BASE),
        embedding(weights.embed_tokens()),
        final_norm(weights.model_norm(), RMS_NORM_EPSILON),
        lm_head(weights.lm_head()),
        hidden_dim(weights.model_norm().shape[0]),
        vocab_size(weights.embed_tokens().shape[0]) {
    blocks.reserve(runtime::ModelWeightsGPU::kNumLayers);
    for (size_t layer_idx = 0; layer_idx < runtime::ModelWeightsGPU::kNumLayers; ++layer_idx) {
      blocks.emplace_back(weights.layer(layer_idx), rope_tables, kNumQHeads, kNumKVHeads,
                          kHeadDim);
    }
  }
};

class ReferenceHarness {
 public:
  runtime::CudaContext& context() { return context_; }
  runtime::Workspace& workspace() { return workspace_; }
  ModelState& model() {
    if (!model_state_)
      model_state_ = std::make_unique<ModelState>();
    return *model_state_;
  }

 private:
  runtime::CudaContext context_;
  runtime::Workspace workspace_;
  std::unique_ptr<ModelState> model_state_;
};

bool run_matmul_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                     const string& label) {
  const auto& input_a = fixture.tensor("input_a");
  const auto& input_b = fixture.tensor("input_b");
  const auto& expected = fixture.tensor("expected");

  MatmulOp op;
  auto a_buffer = upload_f32(input_a.f32_data, harness.context());
  auto b_buffer = upload_f32(input_b.f32_data, harness.context());
  runtime::DeviceBuffer<float> c_buffer(expected.f32_data.size());

  op.forward(harness.context(), const_view(a_buffer.view(input_a.shape)),
             const_view(b_buffer.view(input_b.shape)),
             c_buffer.view(expected.shape));

  return check_max_abs(label, download_f32(c_buffer, expected.f32_data.size(), harness.context()),
                       expected.f32_data, kModelTolerance);
}

bool run_residual_add_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                           const string& label) {
  const auto& input = fixture.tensor("input");
  const auto& residual = fixture.tensor("residual");
  const auto& expected = fixture.tensor("expected");

  ResidualAddOp op;
  auto input_buffer = upload_f32(input.f32_data, harness.context());
  auto residual_buffer = upload_f32(residual.f32_data, harness.context());
  runtime::DeviceBuffer<float> output_buffer(expected.f32_data.size());

  op.forward(harness.context(), const_view(input_buffer.view(input.shape)),
             const_view(residual_buffer.view(residual.shape)),
             output_buffer.view(expected.shape));

  return check_max_abs(label,
                       download_f32(output_buffer, expected.f32_data.size(), harness.context()),
                       expected.f32_data, kSyntheticTolerance);
}

bool run_argmax_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                     const string& label) {
  const auto& input = fixture.tensor("input");
  const auto& expected = fixture.tensor("expected");

  ArgmaxOp op;
  auto input_buffer = upload_f32(input.f32_data, harness.context());
  runtime::DeviceBuffer<int> output_buffer(expected.i32_data.size());

  op.forward(harness.context(), const_view(input_buffer.view(input.shape)),
             output_buffer.view({expected.i32_data.size()}));

  vector<int> expected_host(expected.i32_data.begin(), expected.i32_data.end());
  return check_exact_i32(label, download_i32(output_buffer, expected_host.size(), harness.context()),
                         expected_host);
}

bool run_rope_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                   const string& label) {
  const auto& q = fixture.tensor("q");
  const auto& k = fixture.tensor("k");
  const auto& cos = fixture.tensor("cos");
  const auto& sin = fixture.tensor("sin");
  const auto& expected_q = fixture.tensor("expected_q");
  const auto& expected_k = fixture.tensor("expected_k");

  const size_t num_q_heads = static_cast<size_t>(fixture.scalar_i32("num_q_heads"));
  const size_t num_kv_heads = static_cast<size_t>(fixture.scalar_i32("num_kv_heads"));
  const size_t head_dim = static_cast<size_t>(fixture.scalar_i32("head_dim"));
  const size_t position_offset =
      fixture.has("position_offset") ? static_cast<size_t>(fixture.scalar_i32("position_offset")) : 0;

  auto q_buffer = upload_f32(q.f32_data, harness.context());
  auto k_buffer = upload_f32(k.f32_data, harness.context());
  auto cos_buffer = upload_f32(cos.f32_data, harness.context());
  auto sin_buffer = upload_f32(sin.f32_data, harness.context());

  RoPEOp op(const_view(cos_buffer.view(cos.shape)), const_view(sin_buffer.view(sin.shape)),
            num_q_heads, num_kv_heads,
            head_dim);
  op.forward(harness.context(), q_buffer.view(q.shape), k_buffer.view(k.shape), position_offset);

  const bool q_ok = check_max_abs(
      label + " q", download_f32(q_buffer, expected_q.f32_data.size(), harness.context()),
      expected_q.f32_data, kSyntheticTolerance);
  const bool k_ok = check_max_abs(
      label + " k", download_f32(k_buffer, expected_k.f32_data.size(), harness.context()),
      expected_k.f32_data, kSyntheticTolerance);
  return q_ok && k_ok;
}

bool run_rmsnorm_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                      const string& label) {
  const int layer_idx = fixture.scalar_i32("layer_index");
  const auto& input = fixture.tensor("input");
  const auto& expected = fixture.tensor("expected");

  ModelState& model = harness.model();
  RmsNormOp op(model.weights.layer(static_cast<size_t>(layer_idx)).input_layernorm.view(),
               RMS_NORM_EPSILON);
  auto input_buffer = upload_f32(input.f32_data, model.context);
  runtime::DeviceBuffer<float> output_buffer(expected.f32_data.size());
  op.forward(model.context, const_view(input_buffer.view(input.shape)),
             output_buffer.view(expected.shape));

  return check_max_abs(label, download_f32(output_buffer, expected.f32_data.size(), model.context),
                       expected.f32_data, kModelTolerance);
}

bool run_qkv_projection_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                             const string& label) {
  const int layer_idx = fixture.scalar_i32("layer_index");
  const auto& input = fixture.tensor("input");
  const auto& expected_q = fixture.tensor("expected_q");
  const auto& expected_k = fixture.tensor("expected_k");
  const auto& expected_v = fixture.tensor("expected_v");

  ModelState& model = harness.model();
  const runtime::LayerWeightsGPU& layer = model.weights.layer(static_cast<size_t>(layer_idx));
  QKVProjectionOp op(layer.q_proj.view(), layer.k_proj.view(), layer.v_proj.view());

  auto input_buffer = upload_f32(input.f32_data, model.context);
  runtime::DeviceBuffer<float> q_buffer(expected_q.f32_data.size());
  runtime::DeviceBuffer<float> k_buffer(expected_k.f32_data.size());
  runtime::DeviceBuffer<float> v_buffer(expected_v.f32_data.size());

  op.forward(model.context, const_view(input_buffer.view(input.shape)),
             q_buffer.view(expected_q.shape),
             k_buffer.view(expected_k.shape), v_buffer.view(expected_v.shape));

  const bool q_ok = check_max_abs(label + " q",
                                  download_f32(q_buffer, expected_q.f32_data.size(), model.context),
                                  expected_q.f32_data, kModelTolerance);
  const bool k_ok = check_max_abs(label + " k",
                                  download_f32(k_buffer, expected_k.f32_data.size(), model.context),
                                  expected_k.f32_data, kModelTolerance);
  const bool v_ok = check_max_abs(label + " v",
                                  download_f32(v_buffer, expected_v.f32_data.size(), model.context),
                                  expected_v.f32_data, kModelTolerance);
  return q_ok && k_ok && v_ok;
}

bool run_attention_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                        const string& label) {
  const auto& q = fixture.tensor("q");
  const auto& k = fixture.tensor("k");
  const auto& v = fixture.tensor("v");
  const auto& expected = fixture.tensor("expected");
  const size_t num_q_heads = static_cast<size_t>(fixture.scalar_i32("num_q_heads"));
  const size_t num_kv_heads = static_cast<size_t>(fixture.scalar_i32("num_kv_heads"));
  const size_t head_dim = static_cast<size_t>(fixture.scalar_i32("head_dim"));

  AttentionOp op(num_q_heads, num_kv_heads, head_dim);
  auto q_buffer = upload_f32(q.f32_data, harness.context());
  auto k_buffer = upload_f32(k.f32_data, harness.context());
  auto v_buffer = upload_f32(v.f32_data, harness.context());
  runtime::DeviceBuffer<float> output_buffer(expected.f32_data.size());

  op.forward(harness.context(), harness.workspace(), const_view(q_buffer.view(q.shape)),
             const_view(k_buffer.view(k.shape)), const_view(v_buffer.view(v.shape)),
             output_buffer.view(expected.shape));

  return check_max_abs(label,
                       download_f32(output_buffer, expected.f32_data.size(), harness.context()),
                       expected.f32_data, kModelTolerance);
}

bool run_swiglu_ffn_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                         const string& label) {
  const int layer_idx = fixture.scalar_i32("layer_index");
  const auto& input = fixture.tensor("input");
  const auto& expected = fixture.tensor("expected");

  ModelState& model = harness.model();
  const runtime::LayerWeightsGPU& layer = model.weights.layer(static_cast<size_t>(layer_idx));
  FFNOp op(layer.gate_proj.view(), layer.up_proj.view(), layer.down_proj.view());
  auto input_buffer = upload_f32(input.f32_data, model.context);
  runtime::DeviceBuffer<float> output_buffer(expected.f32_data.size());

  op.forward(model.context, model.workspace, const_view(input_buffer.view(input.shape)),
             output_buffer.view(expected.shape));

  return check_max_abs(label, download_f32(output_buffer, expected.f32_data.size(), model.context),
                       expected.f32_data, kModelTolerance);
}

bool run_decoder_block_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                            const string& label) {
  const int layer_idx = fixture.scalar_i32("layer_index");
  const size_t position_offset =
      fixture.has("position_offset") ? static_cast<size_t>(fixture.scalar_i32("position_offset")) : 0;
  const auto& input = fixture.tensor("input");
  const auto& expected = fixture.tensor("expected");

  ModelState& model = harness.model();
  auto input_buffer = upload_f32(input.f32_data, model.context);
  runtime::DeviceBuffer<float> output_buffer(expected.f32_data.size());

  model.blocks[static_cast<size_t>(layer_idx)].forward(
      model.context, model.workspace, const_view(input_buffer.view(input.shape)),
      output_buffer.view(expected.shape),
      position_offset);

  return check_max_abs(label, download_f32(output_buffer, expected.f32_data.size(), model.context),
                       expected.f32_data, kModelTolerance);
}

bool run_output_layer_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                           const string& label) {
  const auto& input = fixture.tensor("input");
  const auto& expected_logits = fixture.tensor("expected_logits");

  ModelState& model = harness.model();
  auto input_buffer = upload_f32(input.f32_data, model.context);
  runtime::DeviceBuffer<float> norm_buffer(input.f32_data.size());
  runtime::DeviceBuffer<float> logits_buffer(expected_logits.f32_data.size());

  model.final_norm.forward(model.context, const_view(input_buffer.view(input.shape)),
                           norm_buffer.view(input.shape));
  runtime::DeviceTensorView<const float> last_hidden(
      norm_buffer.data() + (input.shape[0] - 1) * input.shape[1], {static_cast<size_t>(1), input.shape[1]});
  model.lm_head.forward(model.context, last_hidden, logits_buffer.view({static_cast<size_t>(1), model.vocab_size}));

  return check_max_abs(label,
                       download_f32(logits_buffer, expected_logits.f32_data.size(), model.context),
                       expected_logits.f32_data, kModelTolerance);
}

vector<float> forward_one_step_logits(ModelState& model, const vector<int>& token_ids) {
  const size_t seq_len = token_ids.size();
  auto& act_a_buffer = model.workspace.get_buffer("reference_act_a", seq_len * model.hidden_dim);
  auto& act_b_buffer = model.workspace.get_buffer("reference_act_b", seq_len * model.hidden_dim);
  runtime::DeviceTensorView<float> act_a = act_a_buffer.view({seq_len, model.hidden_dim});
  runtime::DeviceTensorView<float> act_b = act_b_buffer.view({seq_len, model.hidden_dim});

  model.embedding.forward(model.context, token_ids, act_a);

  runtime::DeviceTensorView<float> current = act_a;
  runtime::DeviceTensorView<float> next = act_b;
  for (const DecoderBlock& block : model.blocks) {
    block.forward(model.context, model.workspace, const_view(current), next, 0);
    std::swap(current, next);
  }

  model.final_norm.forward(model.context, const_view(current), next);
  current = next;

  auto& logits_buffer = model.workspace.get_buffer("reference_logits", model.vocab_size);
  runtime::DeviceTensorView<float> logits = logits_buffer.view({static_cast<size_t>(1), model.vocab_size});
  runtime::DeviceTensorView<const float> last_hidden(
      current.data + (seq_len - 1) * model.hidden_dim, {static_cast<size_t>(1), model.hidden_dim});
  model.lm_head.forward(model.context, last_hidden, logits);

  return download_f32(logits_buffer, model.vocab_size, model.context);
}

bool run_forward_one_step_case(ReferenceHarness& harness, const ReferenceFixture& fixture,
                               const string& label) {
  const auto& token_ids = fixture.tensor("token_ids");
  const auto& expected_logits = fixture.tensor("expected_logits");
  const int expected_next_token = fixture.scalar_i32("expected_next_token");

  vector<int> host_token_ids(token_ids.i32_data.begin(), token_ids.i32_data.end());
  ModelState& model = harness.model();
  const vector<float> logits = forward_one_step_logits(model, host_token_ids);

  const bool logits_ok = check_max_abs(label + " logits", logits, expected_logits.f32_data, kModelTolerance);
  const int got_next_token = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  const bool token_ok = check_exact_i32(label + " next_token", {got_next_token}, {expected_next_token});
  return logits_ok && token_ok;
}

bool run_case(ReferenceHarness& harness, const fs::path& path) {
  const string label = path.stem().string();
  const ReferenceFixture fixture(path.string());

  if (has_prefix(label, "matmul_"))
    return run_matmul_case(harness, fixture, label);
  if (has_prefix(label, "residual_add_"))
    return run_residual_add_case(harness, fixture, label);
  if (has_prefix(label, "argmax_"))
    return run_argmax_case(harness, fixture, label);
  if (has_prefix(label, "rope_"))
    return run_rope_case(harness, fixture, label);
  if (has_prefix(label, "rmsnorm_"))
    return run_rmsnorm_case(harness, fixture, label);
  if (has_prefix(label, "qkv_projection_"))
    return run_qkv_projection_case(harness, fixture, label);
  if (has_prefix(label, "attention_"))
    return run_attention_case(harness, fixture, label);
  if (has_prefix(label, "swiglu_ffn_"))
    return run_swiglu_ffn_case(harness, fixture, label);
  if (has_prefix(label, "decoder_block_"))
    return run_decoder_block_case(harness, fixture, label);
  if (has_prefix(label, "output_layer_"))
    return run_output_layer_case(harness, fixture, label);
  if (has_prefix(label, "forward_one_step_"))
    return run_forward_one_step_case(harness, fixture, label);

  throw runtime_error("reference_tests: unsupported fixture type '" + label + "'");
}

}  // namespace

int main(int argc, char* argv[]) {
  const char* GREEN = "\033[32m";
  const char* RED = "\033[31m";
  const char* RESET = "\033[0m";

  if (!fs::exists(kFixtureDir)) {
    cout << RED << "Fixture directory not found: " << kFixtureDir << RESET << "\n";
    return 1;
  }

  vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(kFixtureDir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bin")
      paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());

  const string filter = argc >= 2 ? argv[1] : "";
  ReferenceHarness harness;
  int failures = 0;
  int ran = 0;

  for (const fs::path& path : paths) {
    const string label = path.stem().string();
    if (!filter.empty() && label.find(filter) == string::npos)
      continue;

    ++ran;
    try {
      const bool ok = run_case(harness, path);
      cout << (ok ? GREEN : RED) << (ok ? "[PASS] " : "[FAIL] ") << label << RESET << "\n";
      if (!ok)
        ++failures;
    } catch (const std::exception& ex) {
      ++failures;
      cout << RED << "[ERROR] " << label << RESET << " " << ex.what() << "\n";
    }
  }

  if (ran == 0) {
    cout << RED << "No fixtures matched in " << kFixtureDir << RESET << "\n";
    return 2;
  }

  cout << (failures == 0 ? GREEN : RED) << "Ran " << ran << " fixture cases, failures="
       << failures << RESET << "\n";
  return failures == 0 ? 0 : 1;
}
