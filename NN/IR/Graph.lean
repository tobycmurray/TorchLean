/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# IR Graph

`NN.IR.Graph` is TorchLean’s canonical *op-tagged* DAG IR.

Today it is used as the shared target for:
- TorchLean to verifier lowering (`NN/Verification/TorchLean/Lowering.lean`),
- bound-propagation / verification tooling (CROWN/LiRPA) (`NN/MLTheory/CROWN/Graph.lean`),
- IR → PyTorch emission (`NN/Runtime/PyTorch/Export/IRPyTorch.lean`),
- compact example graphs (e.g. `NN/Examples/DeepDives/GraphSpec/Tutorial.lean`).

Longer-term, the intent is to use the same IR as a bridge target for:
- spec-level graphs (lower a model spec to an IR graph),
- runtime autograd traces (reify a runtime tape/graph into the same IR),
- verifiers (IBP/CROWN/affine passes) and export tooling.

What this file commits to is the graph *structure* (ops, dependencies, and shapes). Parameter
payloads (weights/bias/const values) live in backend-specific stores keyed by node id. This split
keeps one graph format usable across:

- verification (where parameters often carry additional metadata like bounds or perturbation sets),
- export (where parameters may be emitted as PyTorch `nn.Parameter`s or ONNX initializers),
- and runtime execution/tracing (where parameters may already live in a separate module state).

Like a PyTorch FX graph or TorchScript IR, nodes are operations, edges are data dependencies, and
execution follows topological order. TorchLean additionally attaches explicit *shape* metadata to
every node for verification and proofs.

References / related systems:
- PyTorch FX docs: https://pytorch.org/docs/stable/fx.html
- TorchScript overview: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers as separate parameter store): https://onnx.ai/

## Conventions (important)

- **Topo order**: a node only references parents with smaller ids.
- **Id discipline**: in most builders, `node.id` is expected to equal its index in `Graph.nodes`.
  (For example, TorchLean lowering uses `freshId := nodes.size` and then appends.)
- **External parameters**:
  - `OpKind.const` stores its `valueShape` here, but the constant value is stored externally
    (e.g. in a verifier `ParamStore` keyed by node id).
  - Some ops (notably `OpKind.linear` and `OpKind.conv`) typically use *external* parameter stores
    keyed by node id; in those cases the node’s `parents` array only contains the *runtime inputs*
    (e.g. the activation input `x`), not the weights/bias tensors.

This file does **not** implement evaluation or shape inference. Those live in:
- `NN/IR/Semantics.lean` (evaluation semantics for a chosen scalar backend),
- `NN/IR/Infer.lean` / `NN/IR/Check.lean` (shape inference/checking utilities),
- and backend-specific passes (verification/export) that interpret `OpKind` in their own setting.
-/

@[expose] public section


namespace NN.IR

open Spec

/-- A row-major Boolean mask carried by an IR operation.

The payload records its logical tensor shape separately from the flat array so graph validation can
reject malformed serialized or programmatically constructed masks before evaluation. `true` means
that the corresponding entry is allowed.
-/
structure HardMask where
  shape : Shape
  allowed : Array Bool
  deriving Repr

/-- Per-axis geometry for pooling and convolution operators.

All three lists describe the same spatial suffix of the input tensor. Keeping the geometry in one
record prevents frontends from silently imposing a common stride or padding on every axis.
-/
structure WindowConfig where
  /-- Number of spatial axes. -/
  spatialRank : Nat
  /-- Window extent along each spatial axis. -/
  kernel : Spec.Tensor Nat [spatialRank]
  /-- Step along each spatial axis. -/
  stride : Spec.Tensor Nat [spatialRank]
  /-- Symmetric padding along each spatial axis. -/
  padding : Spec.Tensor Nat [spatialRank]
  deriving Repr

/-- Shape metadata for an arbitrary-dimensional convolution.

Axes before `channelAxis` are preserved and mapped independently. The channel axis is replaced by
`outChannels`; every following axis is spatial and is governed by `window`.
-/
structure ConvConfig extends WindowConfig where
  /-- Spacing between kernel elements along each spatial axis. -/
  dilation : Spec.Tensor Nat [spatialRank] := Spec.Tensor.dim fun _ => Spec.Tensor.scalar 1
  /-- Zero padding after the input along each spatial axis.

  The inherited `padding` field is the padding before the input. Keeping both sides explicit
  represents asymmetric padding without introducing rank-specific convolution variants. -/
  paddingAfter : Spec.Tensor Nat [spatialRank] := padding
  /-- Number of channel groups. `1` is an ordinary dense convolution. -/
  groups : Nat := 1
  /-- Axis containing the input channels. -/
  channelAxis : Nat
  /-- Expected extent of the input-channel axis. -/
  inChannels : Nat
  /-- Extent of the output-channel axis. -/
  outChannels : Nat
  deriving Repr

/-- Operation kinds in an op-tagged computation graph. -/
inductive OpKind where
  | input
      -- Designated graph input (analogous to a PyTorch FX graph input).
  | const (valueShape : Shape)
      -- Constant tensor. We record the shape here, but keep the *value* in an external store
      -- (e.g. verifier parameters, exporter initializers).
  | permute (perm : Array Nat)
      -- Permute axes (0-based). Similar to `torch.permute`.
  | transpose (axis₁ axis₂ : Nat)
      -- Swap two arbitrary axes (0-based). Similar to `torch.transpose`.
  | detach
      -- Identity in the forward pass; marks a gradient stop at runtime (analogous to
      -- `Tensor.detach()`).
  | randUniform (seed : Nat)
      -- Deterministic U[0,1) tensor (seeded). We keep RNG explicit because verification needs a
      -- stable, replayable source of “randomness”.
  | bernoulliMask (seed : Nat)
      -- Deterministic {0,1} mask (seeded); parent is keepProb : scalar.
      -- This is the IR-level representation we use for dropout-style masks.
  | add
      -- Elementwise addition (broadcasting is explicit via `broadcastTo`).
  | sub
      -- Elementwise subtraction.
  | mul_elem
      -- Elementwise multiplication.
  | abs
      -- Elementwise absolute value.
  | sqrt
      -- Elementwise square root (ties to the scalar backend semantics).
  | inv
      -- Elementwise reciprocal (1/x).
  | maxElem
      -- Elementwise max.
  | minElem
      -- Elementwise min.
  | maxPool (config : WindowConfig)
      -- Max pool over the final `config.kernel.length` axes. Leading axes are preserved.
  | avgPool (config : WindowConfig)
      -- Average pool over the same spatial suffix, including zero padding in the divisor.
  | broadcastTo (s₁ s₂ : Shape)
      -- Broadcast parent from s₁→s₂ (analogous to `torch.broadcast_to` / `Tensor.expand`).
  | reduceSum (axis : Nat)
      -- Sum along an axis (axis must be valid).
  | reduceMean (axis : Nat)
      -- Mean along an axis (axis must be valid).
  | sum
      -- Sum reduction to scalar (convenience op used by some loss/verification code paths).
  | matmul
      -- Matrix multiply over the final two axes, preserving a shared leading shape.
  | linear
      -- Affine layer `y = W x + b`. Parameters live in an external store keyed by node id;
      -- the sole parent is the activation input `x`.
  | conv (config : ConvConfig)
      -- Arbitrary-dimensional grouped convolution. Parameters live in an external store keyed by
      -- node id.
  | batchNormEval (channelAxis channels : Nat)
      -- Eval-mode BatchNorm along an explicit channel axis. Affine parameters and running
      -- statistics live in an external store keyed by node id.
  | relu | tanh | sigmoid | exp | log | sin | cos
      -- Common elementwise activations / nonlinearities.
  | minMax (channelAxis : Nat)
      -- MinMax (a.k.a. MaxMin): pair adjacent entries along `channelAxis` and replace each pair
      -- by its minimum and its maximum, smaller first. Anil-Lucas-Grosse, ICML 2019; the standard
      -- activation for Lipschitz-constrained training, where ReLU loses gradient norm.
      -- Not pointwise, and not expressible from the other ops: none of them selects or permutes
      -- *within* an axis. The axis extent must be even.
  | softmax (axis : Nat)
      -- Softmax along an axis.
  | hardMaskedSoftmax (mask : HardMask)
      -- Stable last-axis softmax with exactly zero mass at blocked entries.
  | layernorm (axis : Nat)
      -- LayerNorm over the suffix of dimensions starting at `axis`.
      -- PyTorch analogue: `F.layer_norm(x, normalized_shape=x.shape[axis:])`.
      -- Note: this IR node is the *pure* normalization (gamma=1, beta=0); the common affine form
      -- is typically represented by surrounding `mul_elem`/`add` nodes with broadcasted constants.
  | reshape (inShape outShape : Shape)
      -- Pure reshape (no data movement).
  | flatten (s : Shape)
      -- Flatten to a vector of length `Spec.Shape.size s`.
  | concat (axis : Nat)
      -- Concatenate along an axis (verifier/export may allow an arbitrary number of parents ≥ 2).
  | mseLoss
      -- Scalar mean squared error (used in some training/verification examples).
  deriving Repr

/-- Permitted parent-count interval for an IR operation. -/
structure ParentArity where
  /-- Minimum number of parents required by the operation. -/
  min : Nat
  /-- Maximum number of parents, or `none` when no finite upper bound is imposed. -/
  max? : Option Nat
  deriving DecidableEq, Repr

/-- Structural metadata shared by all instances of an IR operation kind. -/
structure OpMetadata where
  /-- Short stable tag used in diagnostics and graph serialization. -/
  tag : String
  /-- Permitted number of parent nodes. -/
  arity : ParentArity
  deriving DecidableEq, Repr

namespace OpKind

/--
Canonical structural metadata for an `OpKind`.

The arity is a *structural* convention only. For example, `linear` has arity 1 because weights and
biases are stored externally and keyed by node id; the graph records only its activation input.
-/
def metadata : OpKind → OpMetadata
  | .input => ⟨"input", ⟨0, some 0⟩⟩
  | .const .. => ⟨"const", ⟨0, some 0⟩⟩
  | .permute .. => ⟨"permute", ⟨1, some 1⟩⟩
  | .transpose .. => ⟨"transpose", ⟨1, some 1⟩⟩
  | .detach => ⟨"detach", ⟨1, some 1⟩⟩
  | .randUniform .. => ⟨"rand_uniform", ⟨0, some 0⟩⟩
  | .bernoulliMask .. => ⟨"bernoulli_mask", ⟨1, some 1⟩⟩
  | .add => ⟨"add", ⟨2, some 2⟩⟩
  | .sub => ⟨"sub", ⟨2, some 2⟩⟩
  | .mul_elem => ⟨"mul_elem", ⟨2, some 2⟩⟩
  | .abs => ⟨"abs", ⟨1, some 1⟩⟩
  | .sqrt => ⟨"sqrt", ⟨1, some 1⟩⟩
  | .inv => ⟨"inv", ⟨1, some 1⟩⟩
  | .maxElem => ⟨"max_elem", ⟨2, some 2⟩⟩
  | .minElem => ⟨"min_elem", ⟨2, some 2⟩⟩
  | .maxPool .. => ⟨"max_pool", ⟨1, some 1⟩⟩
  | .avgPool .. => ⟨"avg_pool", ⟨1, some 1⟩⟩
  | .broadcastTo .. => ⟨"broadcastTo", ⟨1, some 1⟩⟩
  | .reduceSum .. => ⟨"reduce_sum", ⟨1, some 1⟩⟩
  | .reduceMean .. => ⟨"reduce_mean", ⟨1, some 1⟩⟩
  | .sum => ⟨"sum", ⟨1, some 1⟩⟩
  | .matmul => ⟨"matmul", ⟨2, some 2⟩⟩
  | .linear => ⟨"linear", ⟨1, some 1⟩⟩
  | .conv .. => ⟨"conv", ⟨1, some 1⟩⟩
  | .batchNormEval .. => ⟨"batch_norm_eval", ⟨1, some 1⟩⟩
  | .minMax .. => ⟨"min_max", ⟨1, some 1⟩⟩
  | .relu => ⟨"relu", ⟨1, some 1⟩⟩
  | .tanh => ⟨"tanh", ⟨1, some 1⟩⟩
  | .sigmoid => ⟨"sigmoid", ⟨1, some 1⟩⟩
  | .exp => ⟨"exp", ⟨1, some 1⟩⟩
  | .log => ⟨"log", ⟨1, some 1⟩⟩
  | .sin => ⟨"sin", ⟨1, some 1⟩⟩
  | .cos => ⟨"cos", ⟨1, some 1⟩⟩
  | .softmax .. => ⟨"softmax", ⟨1, some 1⟩⟩
  | .hardMaskedSoftmax .. => ⟨"hard_masked_softmax", ⟨1, some 1⟩⟩
  | .layernorm .. => ⟨"layernorm", ⟨1, some 1⟩⟩
  | .reshape .. => ⟨"reshape", ⟨1, some 1⟩⟩
  | .flatten .. => ⟨"flatten", ⟨1, some 1⟩⟩
  | .concat .. => ⟨"concat", ⟨2, none⟩⟩
  | .mseLoss => ⟨"mse_loss", ⟨2, some 2⟩⟩

/--
The minimum number of parent nodes expected by an `OpKind`.
-/
def minParents (kind : OpKind) : Nat := kind.metadata.arity.min

/--
An optional maximum number of parent nodes expected by an `OpKind`.

For `concat`, the verifier permits an arbitrary number of inputs (at least 2), so this returns
`none`.
-/
def maxParents? (kind : OpKind) : Option Nat := kind.metadata.arity.max?

/-- A short tag for error messages and debugging output. -/
def tag (kind : OpKind) : String := kind.metadata.tag

/--
Human-facing operation description including operation-local parameters.

`tag` is short and stable for grouping/log filtering. `describe` is for diagnostics:
it prints axes, shapes, seeds, and convolution/pooling metadata so malformed graph dumps are useful
without cross-referencing the original builder.
-/
def describe : OpKind → String
  | .input => "input"
  | .const valueShape => s!"const(shape={repr valueShape})"
  | .permute perm => s!"permute(perm={repr perm})"
  | .transpose axis₁ axis₂ => s!"transpose(axis1={axis₁}, axis2={axis₂})"
  | .detach => "detach"
  | .randUniform seed => s!"rand_uniform(seed={seed})"
  | .bernoulliMask seed => s!"bernoulli_mask(seed={seed})"
  | .add => "add"
  | .sub => "sub"
  | .mul_elem => "mul_elem"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .inv => "inv"
  | .maxElem => "max_elem"
  | .minElem => "min_elem"
  | .maxPool config => s!"max_pool(config={repr config})"
  | .avgPool config => s!"avg_pool(config={repr config})"
  | .broadcastTo s₁ s₂ => s!"broadcastTo(from={repr s₁}, to={repr s₂})"
  | .reduceSum axis => s!"reduce_sum(axis={axis})"
  | .reduceMean axis => s!"reduce_mean(axis={axis})"
  | .sum => "sum"
  | .matmul => "matmul"
  | .linear => "linear(payload=node_id)"
  | .conv config => s!"conv(config={repr config}, payload=node_id)"
  | .batchNormEval channelAxis channels =>
      s!"batch_norm_eval(channelAxis={channelAxis}, channels={channels}, payload=node_id)"
  | .minMax channelAxis => s!"min_max(channelAxis={channelAxis})"
  | .relu => "relu"
  | .tanh => "tanh"
  | .sigmoid => "sigmoid"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .softmax axis => s!"softmax(axis={axis})"
  | .hardMaskedSoftmax mask =>
      s!"hard_masked_softmax(maskShape={repr mask.shape})"
  | .layernorm axis => s!"layernorm(axis={axis})"
  | .reshape inShape outShape => s!"reshape(from={repr inShape}, to={repr outShape})"
  | .flatten s => s!"flatten(shape={repr s})"
  | .concat axis => s!"concat(axis={axis})"
  | .mseLoss => "mse_loss"

end OpKind

/-- Node in the graph. Edges are implicit via parent indices. -/
structure Node where
  /-- Node id. By convention this is also the node's index in `Graph.nodes`. -/
  id       : Nat
  /-- Parent node ids, i.e. data dependencies. Each parent must be smaller than `id`. -/
  parents  : Array Nat
  /-- Operation tag and any operation-local metadata. -/
  kind     : OpKind
  /-- Declared output shape. `NN.IR.Infer` can recompute/check this from parents. -/
  outShape : Shape
  deriving Repr

namespace Node

/-- Check the basic parent-count convention for this node kind. -/
def hasValidArity (n : Node) : Bool :=
  let p := n.parents.size
  match n.kind.maxParents? with
  | some hi => (n.kind.minParents ≤ p) && (p ≤ hi)
  | none => (n.kind.minParents ≤ p)

/--
Check that every parent id is strictly smaller than this node id (topological order).

This is the single most important invariant for the IR:
- it guarantees acyclicity,
- it makes evaluation/inference a simple left-to-right pass,
- and it makes backends predictable (no hidden recursion or “graph rewriting during execution”).
-/
def parentsBelow (n : Node) : Bool :=
  n.parents.all (fun pid => pid < n.id)

/-- Render a compact, user-facing summary (useful in error messages). -/
def summary (n : Node) : String :=
  s!"Node(id={n.id}, kind={n.kind.describe}, parents={n.parents}, outShape={repr n.outShape})"

end Node

/-- Return the sole parent id when an IR node has unary arity. -/
def unaryParent? (parents : Array Nat) : Option Nat :=
  if parents.size = 1 then parents[0]? else none

/-- Return both parent ids when an IR node has binary arity. -/
def binaryParents? (parents : Array Nat) : Option (Nat × Nat) :=
  if parents.size = 2 then some (parents[0]!, parents[1]!) else none

/-- The parent returned by the unary decoder belongs to the source array. -/
theorem mem_of_unaryParent?_eq_some {parents : Array Nat} {parent : Nat}
    (h : unaryParent? parents = some parent) : parent ∈ parents := by
  simp only [unaryParent?] at h
  split at h
  next => exact Array.mem_iff_getElem?.2 ⟨0, h⟩
  next => simp_all

/-- The first parent returned by the binary decoder belongs to the source array. -/
theorem fst_mem_of_binaryParents?_eq_some {parents : Array Nat} {left right : Nat}
    (h : binaryParents? parents = some (left, right)) : left ∈ parents := by
  simp only [binaryParents?] at h
  split at h
  next hsize =>
    have hp : (parents[0]!, parents[1]!) = (left, right) := Option.some.inj h
    have hleft : parents[0]! = left := by simpa using congrArg Prod.fst hp
    have hzero : 0 < parents.size := by simp [hsize]
    have hleft' : parents[0] = left := by simpa [getElem!_pos parents 0 hzero] using hleft
    rw [← hleft']
    exact Array.getElem_mem hzero
  next => simp_all

/-- The second parent returned by the binary decoder belongs to the source array. -/
theorem snd_mem_of_binaryParents?_eq_some {parents : Array Nat} {left right : Nat}
    (h : binaryParents? parents = some (left, right)) : right ∈ parents := by
  simp only [binaryParents?] at h
  split at h
  next hsize =>
    have hp : (parents[0]!, parents[1]!) = (left, right) := Option.some.inj h
    have hright : parents[1]! = right := by simpa using congrArg Prod.snd hp
    have hone : 1 < parents.size := by simp [hsize]
    have hright' : parents[1] = right := by simpa [getElem!_pos parents 1 hone] using hright
    rw [← hright']
    exact Array.getElem_mem hone
  next => simp_all

/-- Entire graph as an array of nodes. Parents must have smaller ids (topo order). -/
structure Graph where
  /-- nodes. -/
  nodes : Array Node
  deriving Repr

namespace Graph

/-- Number of nodes in the graph. -/
def size (g : Graph) : Nat :=
  g.nodes.size

/-- Safe node lookup by id (treating ids as array indices). -/
def getNode? (g : Graph) (id : Nat) : Option Node :=
  g.nodes[id]?

/--
Total node lookup that enforces the common "id discipline" invariant
$\mathrm{nodes}[i].\mathrm{id}=i$.

This is convenient for backends that treat node ids as array indices (verifiers, exporters, pretty
printers). The error message is meant to point to a builder bug rather than a user error.
-/
def getNode (g : Graph) (id : Nat) : Except String Node := do
  match g.getNode? id with
  | none => throw s!"IR graph: node id out of bounds: {id}"
  | some n =>
      if n.id != id then
        throw s!"IR graph: internal error: nodes[{id}].id = {n.id} (expected {id})"
      pure n

/-- A successful checked lookup returns a node whose stored id is the requested array index. -/
theorem getNode_id_eq {g : Graph} {id : Nat} {node : Node}
    (h : g.getNode id = .ok node) : node.id = id := by
  unfold getNode at h
  split at h
  · contradiction
  · rename_i found hFound
    split at h
    · contradiction
    · rename_i hId
      have hn : found.id = id := by simpa using hId
      change Except.ok found = Except.ok node at h
      injection h with hNode
      subst node
      exact hn

/-- Safe outShape lookup by id. -/
def outShape? (g : Graph) (id : Nat) : Option Shape :=
  (getNode? g id).map (·.outShape)

/--
Explain why `Node.hasValidArity` failed.

This returns a human-facing message rather than structured data; callers use it for diagnostics.
-/
def arityError (n : Node) : String :=
  let got := n.parents.size
  match n.kind.maxParents? with
  | some hi =>
      s!"bad parent count for {n.kind.tag}: expected {n.kind.minParents}..{hi}, got {got}"
  | none =>
      s!"bad parent count for {n.kind.tag}: expected at least {n.kind.minParents}, got {got}"

/--
Basic well-formedness check used by verifier code paths.

This checks:
- node ids match array indices (common construction invariant),
- each node respects its op arity convention, and
- all parent ids are strictly smaller than the node id (topological order).

We keep this as a boolean predicate because some passes want a fast “yes/no” filter. If you need a
human-facing error, use `checkWellFormed`.
-/
def wellFormed (g : Graph) : Bool :=
  (List.finRange g.nodes.size).all (fun i =>
    match g.nodes[i]? with
    | none => false
    | some n => (n.id = i) && n.hasValidArity && n.parentsBelow)

/--
Like `wellFormed`, but returns a helpful error message on failure.

This is useful when you want a *clean* user error rather than a silent `false`.
-/
def checkWellFormed (g : Graph) : Except String Unit := do
  for i in [0:g.nodes.size] do
    match g.nodes[i]? with
    | none =>
        throw s!"IR graph: internal error: missing node at index {i}"
    | some n =>
        if n.id != i then
          throw s!"IR graph: id discipline violated at index {i}: nodes[{i}].id = {n.id}"
        if !n.hasValidArity then
          throw s!"IR graph: node {i}: {arityError n} ({n.summary})"
        -- Because we have `n.id = i`, checking `pid < n.id` also implies `pid` is in-bounds.
        for pid in n.parents do
          if pid ≥ n.id then
            throw s!"IR graph: node {i}: parent id {pid} is not < {n.id} ({n.summary})"

end Graph

/-- Default node used only to satisfy generic container APIs; real graphs should not rely on it. -/
instance : Inhabited Node where
  default := { id := 0, parents := #[], kind := OpKind.input, outShape := Shape.scalar }

end NN.IR
