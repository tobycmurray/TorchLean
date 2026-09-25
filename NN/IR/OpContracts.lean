/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Spec.Layers.Pooling

/-!
# Operation Contracts

Shared operation contracts for `NN.IR.Graph`.

Several IR passes need to agree on the same small set of “shape contracts”:

- `NN.IR.Infer`: recompute output shapes from op parameters + parent shapes.
- `NN.IR.Check`: expose the documented `Graph.checkShapes` wrapper.
- `NN.IR.Semantics`: evaluate nodes and reject ill-shaped graphs with readable error messages.

The point of this file is to keep shape arithmetic out of individual passes. If an op has nontrivial
shape behavior (concat, matmul, pooling, convolution, LayerNorm flattening, axis moves), define the
contract here first and call it from inference/semantics instead of copying the formula.
-/

@[expose] public section

namespace NN.IR

open _root_.Spec

/-!
## Small shape utilities

These helpers are used by multiple IR passes, especially `Infer` and `Semantics`.
-/

namespace ShapeUtil

/-- The output shape of flattening a tensor of shape `s` to a 1D vector. -/
def flattenOutShape (s : Shape) : Shape :=
  .dim (Spec.Shape.size s) .scalar

end ShapeUtil

namespace OpContracts

/-!
## Generic contract helpers

These functions live outside any particular pass (`Infer`/`Check`/`Semantics`) so they can be
reused without introducing import cycles.
-/

/-- Check that an `axis` is in-bounds for a given shape. -/
def checkAxisValid (axis : Nat) (s : Shape) : Except String Unit := do
  if axis < Spec.Shape.rank s then
    pure ()
  else
    throw s!"invalid axis {axis} for rank {Spec.Shape.rank s}"

/-- Check that the axis a MinMax pairs along exists and has even extent.

An odd extent would leave one entry unpaired.  `Activation.minMaxOuterSpec` is
total there — it leaves the entry alone — but a network whose channel count is
odd is a modelling mistake rather than something to interpret silently, so the
graph is rejected. -/
def checkMinMaxAxis (axis : Nat) (s : Shape) : Except String Unit := do
  let _ ← checkAxisValid axis s
  match (Spec.Shape.toList s)[axis]? with
  | none => throw s!"min_max: invalid axis {axis} for rank {Spec.Shape.rank s}"
  | some extent =>
      if extent % 2 = 0 then
        pure ()
      else
        throw s!"min_max: axis {axis} has odd extent {extent}; MinMax pairs entries up"

/-- Check that a natural-number op parameter is nonzero. -/
def checkPositive (tag param : String) (n : Nat) : Except String Unit := do
  if n = 0 then
    throw s!"{tag}: {param} must be > 0"
  else
    pure ()

/-- Axis permutation that swaps `axis₁` and `axis₂` and leaves every other axis fixed. -/
def transposePerm (rank axis₁ axis₂ : Nat) : Except String (Array Nat) := do
  if axis₁ ≥ rank then
    throw s!"transpose: axis {axis₁} out of range for rank {rank}"
  if axis₂ ≥ rank then
    throw s!"transpose: axis {axis₂} out of range for rank {rank}"
  pure <| (Array.range rank).map fun axis =>
    if axis = axis₁ then axis₂ else if axis = axis₂ then axis₁ else axis

/-- Infer the shape obtained by swapping two arbitrary axes. -/
def inferTransposeOutShape (axis₁ axis₂ : Nat) (parent : Shape) : Except String Shape := do
  let perm ← transposePerm (Spec.Shape.rank parent) axis₁ axis₂
  match Spec.Shape.permute? parent perm.toList with
  | some output => pure output
  | none => throw s!"transpose: internal invalid permutation {repr perm} for {repr parent}"

/--
Reconstruct the proof object required by the typed tensor broadcast primitive.

IR nodes store dynamic shapes, so every pass that accepts `.broadcastTo` must rebuild this witness
instead of trusting that the declared input and output shapes are compatible.
-/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | s₁, s₂ =>
      if hlt : Spec.Shape.rank s₁ < Spec.Shape.rank s₂ then
        match s₂ with
        | .scalar => none
        | .dim n₂ t₂ =>
            (mkCanBroadcastTo? s₁ t₂).map fun tail =>
              Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := s₁) (s₂ := t₂) tail
      else if hgt : Spec.Shape.rank s₂ < Spec.Shape.rank s₁ then
        none
      else
        match s₁, s₂ with
        | .scalar, .scalar => some .scalar
        | .dim n₁ t₁, .dim n₂ t₂ =>
            letI : Shape.SameRank t₁ t₂ := ⟨by
              apply Nat.le_antisymm
              · exact Nat.le_of_not_gt (by simpa [Spec.Shape.rank] using hgt)
              · exact Nat.le_of_not_gt (by simpa [Spec.Shape.rank] using hlt)⟩
            if hEq : n₁ = n₂ then
              (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
                hEq ▸ Shape.CanBroadcastTo.dim_eq (n := n₁) (s₁ := t₁) (s₂ := t₂) tail)
            else if h1 : n₁ = 1 then
              (mkCanBroadcastTo? t₁ t₂).map (fun tail =>
                h1 ▸ Shape.CanBroadcastTo.dim_1_to_n (n := n₂) (s₁ := t₁) (s₂ := t₂) tail)
            else
              none
        | _, _ => none

/--
Compute the matrix dimensions used to interpret `layernorm axis`.

TorchLean’s IR stores LayerNorm as an `axis : Nat` instead of a full `normalized_shape` tuple.
We interpret this in the same way the PyTorch exporter does:

`normalized_shape = dims.drop axis`

That is, we normalize over the **suffix** of dimensions starting at `axis`. To reuse the current
spec primitive (`Spec.layerNorm`), we flatten the input shape `s` into a matrix:

* the row count is the product of dimensions before `axis` (`dims.take axis`);
* the column count is the product of dimensions from `axis` onward (`dims.drop axis`).

LayerNorm then runs over each row and the result is reshaped to `s`. This construction works for
every nonempty tensor rank; the matrix is an evaluation view, not a restriction on the input shape.
-/
def layerNormMatrixDims (axis : Nat) (s : Shape) : Except String (Nat × Nat) := do
  checkAxisValid axis s
  let dims := Shape.toList s
  let seqLen : Nat := (dims.take axis).foldl (fun acc d => acc * d) 1
  let embedDim : Nat := (dims.drop axis).foldl (fun acc d => acc * d) 1
  checkPositive "layernorm" "seqLen" seqLen
  checkPositive "layernorm" "embedDim" embedDim
  pure (seqLen, embedDim)

/--
Compute the inverse of a permutation array.

If `perm` is a permutation of `[0,1,...,r-1]` (where `r = perm.length`), then the inverse `inv`
satisfies $\mathrm{inv}[\mathrm{perm}[i]]=i$.
-/
def inversePerm (perm : Array Nat) : Except String (Array Nat) := do
  let r := perm.size
  let mut inv : Array (Option Nat) := Array.replicate r none
  let mut i : Nat := 0
  for p in perm do
    unless p < r do
      throw s!"permute: axis {p} out of range for rank {r} in {repr perm}"
    if inv[p]!.isSome then
      throw s!"permute: duplicate axis {p} in {repr perm}"
    inv := inv.set! p (some i)
    i := i + 1

  let mut out : Array Nat := #[]
  for j in [0:r] do
    match inv[j]! with
    | some idx => out := out.push idx
    | none => throw s!"permute: missing axis {j} in {repr perm}"
  pure out

/--
Permutation (0-based axes) that moves `axis` to the **last** position, preserving the relative
order of the other axes.

Example: rank four with `axis` set to `1` yields `[0,2,3,1]`.
-/
def permMoveAxisToLast (axis : Nat) (s : Shape) : Except String (Array Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| ((Array.range r).filter (· != axis)).push axis

/--
Permutation (0-based axes) that moves `axis` to the **first** position, preserving the relative
order of the other axes.

Example: rank four with `axis` set to `2` yields `[2,0,1,3]`.
-/
def permMoveAxisToFront (axis : Nat) (s : Shape) : Except String (Array Nat) := do
  checkAxisValid axis s
  let r := Spec.Shape.rank s
  pure <| #[axis] ++ (Array.range r).filter (· != axis)

/--
Infer the output shape for `matmul` from the two parent shapes.

Both inputs must have rank at least two and exactly the same leading shape. The final two axes
follow the usual matrix rule: `(...×m×n) · (...×n×p) → (...×m×p)`.
-/
def inferMatmulOutShape (a b : Shape) : Except String Shape := do
  match a.toList.reverse, b.toList.reverse with
  | n :: m :: leadingRev, p :: n' :: leadingRev' =>
      if _hLeading : leadingRev = leadingRev' then
        if _hInner : n = n' then
          pure <| Shape.ofList (leadingRev.reverse ++ [m, p])
        else
          throw s!"matmul: inner dims mismatch: {n} vs {n'}"
      else
        throw s!"matmul: leading dimensions mismatch: {repr a} vs {repr b}"
  | _, _ =>
      throw s!"matmul: expected rank≥2 inputs, got {repr a} and {repr b}"

/-- Merge one concat input into the accumulated dimensions. -/
def mergeConcatDims (axis : Nat) : Nat → List Nat → List Nat → Except String (List Nat)
  | _, [], [] => pure []
  | index, expected :: expecteds, actual :: actuals => do
      let rest ← mergeConcatDims axis (index + 1) expecteds actuals
      if index = axis then
        pure ((expected + actual) :: rest)
      else if actual = expected then
        pure (expected :: rest)
      else
        throw s!"concat: non-axis dim mismatch at axis {index}: expected {expected}, got {actual}"
  | _, _, _ => throw "concat: rank mismatch"

/-- Fold compatible concat inputs into the accumulated output dimensions. -/
def foldConcatDims (axis : Nat) : List Nat → List Shape → Except String (List Nat)
  | dimensions, [] => pure dimensions
  | dimensions, shape :: shapes => do
      let dimensions ← mergeConcatDims axis 0 dimensions shape.toList
      foldConcatDims axis dimensions shapes

/--
Infer the output shape for `concat` from an array of parent shapes.

Every parent must have the same rank, the selected axis must exist, and all dimensions other than
that axis must agree. The selected dimensions are summed. Both the axis and the tensor rank are
arbitrary.
-/
def inferConcatOutShape (axis : Nat) (parents : Array Shape) : Except String Shape := do
  match parents.toList with
  | first :: second :: rest =>
      checkAxisValid axis first
      let dimensions ← foldConcatDims axis first.toList (second :: rest)
      pure (Shape.ofList dimensions)
  | _ => throw s!"concat: expected at least 2 inputs, got {parents.size}"

/-!
## Sliding-window shape arithmetic

Convolution and pooling preserve a leading channel axis, but their admissible padding domains are
not identical. The contracts below share validation and traversal while retaining the correct
output formula for each operation family.
-/

/-- Output length for a 1D sliding-window op without padding:
$\left\lfloor(\mathrm{in}-k)/\mathrm{stride}\right\rfloor+1$. -/
def slideOut (inLen k stride : Nat) : Nat :=
  Shape.slidingWindowOutDim inLen k stride 0

/-- Output length for a 1D sliding-window op with symmetric padding: `⌊(in + 2*pad - k)/stride⌋ +
  1`. -/
def slideOutPad (inLen k stride padding : Nat) : Nat :=
  Shape.slidingWindowOutDim inLen k stride padding

/-- Effective kernel width for a dilated window. -/
def effectiveKernel (kernel dilation : Nat) : Nat :=
  if kernel = 0 then 0 else dilation * (kernel - 1) + 1

/-- Output length for a dilated window with independent low/high padding. -/
def slideOutDilated (input kernel stride dilation paddingBefore paddingAfter : Nat) : Nat :=
  let effective := effectiveKernel kernel dilation
  let padded := input + paddingBefore + paddingAfter
  if effective = 0 || stride = 0 || padded < effective then
    0
  else
    (padded - effective) / stride + 1

/-- Infer dilated convolution dimensions from one parameter per spatial axis. -/
def inferConvDims (tag : String) (axisNames : List String)
    (inputs kernels strides dilations paddingBefore paddingAfter : List Nat) :
    Except String (List Nat) :=
  go axisNames inputs kernels strides dilations paddingBefore paddingAfter
where
  go : List String → List Nat → List Nat → List Nat → List Nat → List Nat →
      List Nat → Except String (List Nat)
    | [], [], [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        dilation :: dilations, low :: lows, high :: highs => do
      checkPositive tag s!"{axis} kernel" kernel
      checkPositive tag s!"{axis} stride" stride
      checkPositive tag s!"{axis} dilation" dilation
      let effective := effectiveKernel kernel dilation
      let padded := input + low + high
      if padded < effective then
        throw <|
          s!"{tag}: {axis} window does not fit padded input: input={input}, " ++
            s!"padding=({low}, {high}), effectiveKernel={effective}"
      let rest ← go axes inputs kernels strides dilations lows highs
      pure (slideOutDilated input kernel stride dilation low high :: rest)
    | _, _, _, _, _, _, _ =>
      throw s!"{tag}: spatial metadata ranks must agree"

/--
Reject sliding-window shapes where the kernel has no valid placement.

Lean `Nat` subtraction saturates at zero, so $\mathrm{in}+2\,\mathrm{pad}-k$ would otherwise turn an invalid
window into a plausible one-element output.
-/
def checkWindowFits (tag axis : String) (inLen k padding : Nat) : Except String Unit := do
  let padded := inLen + 2 * padding
  if padded < k then
    throw s!"{tag}: {axis} window does not fit padded input: input={inLen}, padding={padding}, kernel={k}"
  else
    pure ()

/--
Infer the output lengths of a channel-first sliding-window operation.

The four lists describe the input length, kernel width, stride, and symmetric padding on each
spatial axis. Their lengths must agree. Invalid kernels, strides, and windows are rejected before
`Nat` subtraction can hide the error by saturating at zero.
-/
def inferSlidingWindowDims (tag : String) (axisNames : List String)
    (inputs kernels strides paddings : List Nat) : Except String (List Nat) :=
  go axisNames inputs kernels strides paddings
where
  go : List String → List Nat → List Nat → List Nat → List Nat → Except String (List Nat)
    | [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        padding :: paddings => do
        checkPositive tag s!"{axis} kernel" kernel
        checkPositive tag s!"{axis} stride" stride
        checkWindowFits tag axis input kernel padding
        let rest ← go axes inputs kernels strides paddings
        pure (slideOutPad input kernel stride padding :: rest)
    | _, _, _, _, _ =>
        throw s!"{tag}: axis-name, input, kernel, stride, and padding ranks must agree"

/--
Infer pooling output lengths while enforcing the same basic window checks as graph validation.

Pooling uses `poolOutDim`, which assigns an empty output to an empty input axis, an oversized
window, or padding outside the pooling domain. Kernels and strides must still be positive.
-/
def inferPoolingDims (tag : String) (axisNames : List String)
    (inputs kernels strides paddings : List Nat) : Except String (List Nat) :=
  go axisNames inputs kernels strides paddings
where
  go : List String → List Nat → List Nat → List Nat → List Nat → Except String (List Nat)
    | [], [], [], [], [] => pure []
    | axis :: axes, input :: inputs, kernel :: kernels, stride :: strides,
        padding :: paddings => do
        checkPositive tag s!"{axis} kernel" kernel
        checkPositive tag s!"{axis} stride" stride
        let rest ← go axes inputs kernels strides paddings
        pure (poolOutDim input kernel stride padding :: rest)
    | _, _, _, _, _ =>
        throw s!"{tag}: axis-name, input, kernel, stride, and padding ranks must agree"

/-- Validated shape information for pooling over a tensor suffix.

The plan is the common boundary between shape inference, denotational evaluation, and executable
lowering. It carries the exact split and positivity evidence required by the typed pooling
operators, so those passes cannot silently disagree about which axes are spatial. -/
structure PoolPlan (config : WindowConfig) (parent : Shape) where
  /-- Axes preserved before the pooled suffix. -/
  leading : Shape
  /-- Input extents along the pooled axes. -/
  spatial : Spec.Tensor Nat [config.spatialRank]
  /-- The prefix and spatial suffix reconstruct the parent shape. -/
  concat_eq : leading.concat (Shape.ofList spatial.toList) = parent
  /-- Every kernel extent is nonzero. -/
  kernelNonzero : ∀ axis : Fin config.spatialRank, config.kernel.getScalar axis ≠ 0
  /-- Every stride is nonzero. -/
  strideNonzero : ∀ axis : Fin config.spatialRank, config.stride.getScalar axis ≠ 0

namespace PoolPlan

/-- Output shape computed by a validated pooling plan. -/
def outShape {config : WindowConfig} {parent : Shape} (plan : PoolPlan config parent) : Shape :=
  plan.leading.concat <|
    Shape.ofList
      (Spec.poolOutSpatialPad plan.spatial config.kernel config.stride config.padding).toList

end PoolPlan

/-- Validate and plan pooling over a spatial suffix of arbitrary rank.

Every preceding axis is preserved, so one operation handles unbatched tensors, ordinary batches,
and tensors with several leading batch dimensions.
-/
def planPool (tag : String) (config : WindowConfig) (parent : Shape) :
    Except String (PoolPlan config parent) := do
  checkPositive tag "spatial rank" config.spatialRank
  if hRank : config.spatialRank ≤ parent.rank then
    let split := Shape.splitSuffix parent config.spatialRank hRank
    let spatial := Tensor.ofArrayExact split.suffix.toArray (by simpa using split.suffix_length)
    if hKernel : ∀ axis : Fin config.spatialRank, config.kernel.getScalar axis ≠ 0 then
      if hStride : ∀ axis : Fin config.spatialRank, config.stride.getScalar axis ≠ 0 then
        let leadingRank := split.leading.rank
        let axisNames :=
          (List.range config.spatialRank).map fun axis => s!"axis {leadingRank + axis}"
        let _ ← inferPoolingDims tag axisNames split.suffix config.kernel.toList
          config.stride.toList config.padding.toList
        pure
          { leading := split.leading
            spatial := spatial
            concat_eq := by simpa [spatial] using split.concat_eq
            kernelNonzero := hKernel
            strideNonzero := hStride }
      else
        throw s!"{tag}: stride extents must be nonzero"
    else
      throw s!"{tag}: kernel extents must be nonzero"
  else
    throw s!"{tag}: kernel rank {config.spatialRank} exceeds input rank {parent.rank}"

/-- Infer the output shape of an arbitrary-rank pooling operation. -/
def inferPoolOutShape (tag : String) {spatialRank : Nat}
    (kernels strides paddings : Spec.Tensor Nat [spatialRank]) (parent : Shape) : Except String Shape := do
  let config : WindowConfig :=
    { spatialRank := spatialRank, kernel := kernels, stride := strides, padding := paddings }
  let plan ← planPool tag config parent
  pure plan.outShape

/-- Infer the output shape for the full parameterized convolution configuration. -/
def inferConvConfigOutShape (tag : String) (config : ConvConfig) (parent : Shape) :
    Except String Shape := do
  checkPositive tag "inChannels" config.inChannels
  checkPositive tag "outChannels" config.outChannels
  checkPositive tag "groups" config.groups
  if config.inChannels % config.groups != 0 then
    throw s!"{tag}: in_channels={config.inChannels} is not divisible by groups={config.groups}"
  if config.outChannels % config.groups != 0 then
    throw s!"{tag}: out_channels={config.outChannels} is not divisible by groups={config.groups}"
  checkAxisValid config.channelAxis parent
  let dims := parent.toList
  let some actualChannels := dims[config.channelAxis]?
    | throw s!"{tag}: internal missing channel axis {config.channelAxis} in {repr parent}"
  if actualChannels != config.inChannels then
    throw s!"{tag}: input-channel mismatch: op={config.inChannels} vs input={actualChannels}"
  let inputs := dims.drop (config.channelAxis + 1)
  if inputs.length != config.spatialRank then
    throw <|
      s!"{tag}: kernel rank {config.spatialRank} does not match the {inputs.length} spatial axes " ++
        s!"after channel axis {config.channelAxis}"
  let axisNames :=
    (List.range config.spatialRank).map fun axis => s!"axis {config.channelAxis + 1 + axis}"
  let outputs ← inferConvDims tag axisNames inputs config.kernel.toList
    config.stride.toList config.dilation.toList config.padding.toList config.paddingAfter.toList
  pure (Shape.ofList (dims.take config.channelAxis ++ config.outChannels :: outputs))

/--
Infer dense, unit-dilation convolution output geometry for an arbitrary spatial rank.

This is the ordinary convolution specialization of `inferConvConfigOutShape`; keeping one checker
prevents the default and parameterized APIs from assigning different shapes to the same operation.
-/
def inferConvOutShape (tag : String) (channelAxis inChannels outChannels : Nat)
    {spatialRank : Nat} (kernels strides paddings : Spec.Tensor Nat [spatialRank]) (parent : Shape) :
    Except String Shape :=
  inferConvConfigOutShape tag
    { spatialRank := spatialRank
      kernel := kernels
      stride := strides
      padding := paddings
      dilation := Spec.fill 1 [spatialRank]
      paddingAfter := paddings
      groups := 1
      channelAxis := channelAxis
      inChannels := inChannels
      outChannels := outChannels }
    parent

/-- Check eval-mode BatchNorm metadata against an arbitrary channel axis. -/
def inferBatchNormEvalOutShape (channelAxis channels : Nat) (parent : Shape) : Except String Shape := do
  checkPositive "batch_norm_eval" "channels" channels
  checkAxisValid channelAxis parent
  let some actualChannels := parent.toList[channelAxis]?
    | throw s!"batch_norm_eval: internal missing channel axis {channelAxis} in {repr parent}"
  if channels = actualChannels then
    pure parent
  else
    throw s!"batch_norm_eval: channel mismatch: op={channels} vs input={actualChannels}"

end OpContracts

end NN.IR
