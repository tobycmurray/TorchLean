/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.HardMask
public import NN.IR.Payload
public import NN.Spec.Core.Context
public import NN.Runtime.Autograd.TorchLean.Random
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.Tensor.SomeTensor
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

import NN.Spec.Core.Tensor.Linalg

/-!
# Semantics

Denotational semantics for `NN.IR.Graph`.

This file defines an evaluator for the current IR fragment:
- it evaluates nodes in SSA/topological order,
- each node applies the corresponding *spec-layer* tensor operation to its parents,
- parameter payloads for `const`, `linear`, and `conv` are supplied by an explicit `Payload`.

The evaluator is total on well-formed, well-shaped graphs and returns `Except String` on malformed
graphs or missing payloads.

Softmax and layer norm:
- `softmax axis` normalizes independently along the zero-based tensor dimension named by `axis`.
  The canonical specification handles every in-bounds dimension and preserves the tensor shape.
  This matches the meaning of `torch.softmax(x, dim=axis)` in PyTorch.
- `layernorm axis` matches PyTorch's `F.layer_norm(x, normalized_shape=x.shape[axis:])` convention:
  `axis` selects the start of the **normalized suffix**. We implement this by reshaping the tensor
  into a 2D view `(seqLen, embedDim)`, applying the spec 2D LayerNorm (`Spec.layerNorm`), then
  reshaping back.

How this relates to PyTorch:
- `Graph.nodes` is analogous to a topologically-sorted IR like FX/TorchScript.
- `Payload` is analogous to “parameters / buffers / constants” that live outside the pure graph
  structure.
- The evaluator is a pure, denotational model of running the graph. It is designed for clarity and
  for connecting to proofs and verification passes (not for performance).

References / related systems:
- PyTorch FX: https://pytorch.org/docs/stable/fx.html
- TorchScript: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers): https://onnx.ai/
-/

@[expose] public section


namespace NN.IR

open _root_.Spec
open _root_.Spec.Tensor

namespace Graph

/-! ## Permutation lowering -/

/--
Compute a sequence of adjacent swaps that realizes a target permutation.

This is used to implement `.permute` by repeatedly applying `swapAdjacentAtDepth`, which is already
available in the spec tensor library. If the permutation is ill-formed, this returns an error
explaining what went wrong.
-/
def swapDepthsForPerm (perm : Array Nat) (r : Nat) : Except String (Array Nat) := do
  let mut cur : List Nat := List.range r
  let mut swapsRev : List Nat := []
  for i in [0:r] do
    match perm[i]? with
    | none => throw s!"permute: internal error: missing perm[{i}]"
    | some target =>
        match cur.findIdx? (· == target) with
        | none => throw s!"permute: internal error: target axis {target} not in current axes {cur}"
        | some j =>
            let mut k := j
            while k > i do
              swapsRev := (k - 1) :: swapsRev
              cur := Spec.Shape.swapAdjacentAxes cur (k - 1)
              k := k - 1
  pure swapsRev.reverse.toArray

/--
Permute a shape-tagged tensor according to `perm`.

This checks that `perm` is a valid permutation for the input shape (using `Shape.permute?`), then
lowers it to a sequence of adjacent swaps and applies them to the tensor.
-/
def permuteSomeTensor {α : Type} [Context α]
    (v : Spec.SomeTensor α) (perm : Array Nat) : Except String (Spec.SomeTensor α) := do
  let sIn := v.shape
  match Spec.Shape.permute? sIn perm.toList with
  | none => throw s!"permute: invalid permutation {repr perm} for shape {repr sIn}"
  | some _ =>
      let swaps ← swapDepthsForPerm perm (Spec.Shape.rank sIn)
      pure <| swaps.foldl (fun acc d => Spec.SomeTensor.swapAdjacentAtDepth acc d) v

/-!
## Evaluation helpers

The evaluator itself (`evalAt` / `denoteAll`) is a fold over nodes. These helpers keep the fold
readable:
- `expectShape` checks a value's stored shape against the node's declared `outShape`.
- `evalConst`, `evalLinear`, and `evalConv` fetch and apply external payloads keyed by node id.
-/

/-- Check that a shape-erased tensor has the expected shape and recover its statically typed tensor. -/
def expectShape {α : Type} [Context α] [DecidableEq Shape]
    (expected : Shape) (v : Spec.SomeTensor α) : Except String (Tensor α expected) := do
  if h : v.shape = expected then
    -- transport across the shape equality
    pure (h ▸ v.tensor)
  else
    throw s!"IR eval: shape mismatch: expected {repr expected}, got {repr v.shape}"

/-- Denominator for totalized mean reductions over a dynamic IR shape.

For nonempty shapes this is the real element count. For empty shapes, the mathematical mean is
undefined; the IR is total, so it uses denominator `1` and the empty sum contributes `0`.
-/
def meanDenom (s : Shape) : Nat :=
  if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s

/-- Evaluate MSE loss on two shape-erased tensors after checking that their stored shapes agree. -/
def mseLossSomeTensor {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (yVal tVal : Spec.SomeTensor α) : Except String (Spec.SomeTensor α) := do
  if h : yVal.shape = tVal.shape then
    let yT : Tensor α yVal.shape := yVal.tensor
    let tT : Tensor α yVal.shape := h.symm ▸ tVal.tensor
    let s := yVal.shape
    let diff := Tensor.subSpec (α := α) yT tT
    let sq := Tensor.mulSpec (α := α) diff diff
    let total : α := Tensor.sumSpec (α := α) sq
    let mean : α := total / (↑(meanDenom s) : α)
    pure (Spec.SomeTensor.mk (α := α) Shape.scalar (Tensor.scalar mean))
  else
    throw <|
      s!"IR eval: node {i}: mse_loss expects equal shapes, got " ++
        s!"{repr yVal.shape} vs {repr tVal.shape}"

@[simp] theorem mseLossSomeTensor_mk {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) {s : Shape} (y t : Tensor α s) :
    mseLossSomeTensor (α := α) i (Spec.SomeTensor.mk (α := α) s y) (Spec.SomeTensor.mk (α := α) s t) =
      .ok (Spec.SomeTensor.mk (α := α) Shape.scalar
        (Tensor.scalar
          (((Tensor.subSpec (α := α) y t).mulSpec (Tensor.subSpec (α := α) y t)).sumSpec /
            (↑(meanDenom s) : α)))) := by
  simp [mseLossSomeTensor]
  rfl

/-- Transport a `Tensor α (dim n scalar)` across an equality `n = n'` (helper for payload casts). -/
def castDimScalar {α : Type} [Context α] {n n' : Nat}
    (h : n = n') (t : Tensor α [n]) : Tensor α [n'] :=
      by
  simpa [h] using t

/-- Apply one affine map independently at every index of an arbitrary leading shape. -/
def linearLeading {α : Type} [Context α] (leading : Shape) {inDim outDim : Nat}
    (weights : Tensor α [outDim, inDim]) (bias : Tensor α [outDim])
    (input : Tensor α (leading.concat [inDim])) :
    Tensor α (leading.concat [outDim]) :=
  Tensor.mapEach leading
    (fun sample => Tensor.addSpec (Spec.matVecMulSpec weights sample) bias) input

/-- Multiply matrices independently at every index of a shared leading shape. -/
def matmulLeading {α : Type} [Context α] (leading : Shape) {m n p : Nat}
    (left : Tensor α (leading.concat [m, n]))
    (right : Tensor α (leading.concat [n, p])) :
    Tensor α (leading.concat [m, p]) :=
  Tensor.zipEach leading [m, p] Spec.matMulSpec left right

/--
Evaluate a `const` node from the external payload.

Constants are stored “flat” (1D) for convenience, so we check the flattened length matches
`Spec.Shape.size s` and then `unflatten` to the requested shape.
-/
def evalConst {α : Type} [Context α]
    (payload : Payload α) (id : Nat) (s : Shape) : Except String (Tensor α s) := do
  match payload.const? id with
  | none => throw s!"IR eval: missing const payload for node {id}"
  | some c =>
      if h : c.n = Spec.Shape.size s then
        let v' : Tensor α [Spec.Shape.size s] := castDimScalar (α := α) h c.v
        pure (Tensor.unflattenSpec (α := α) (s := s) v')
      else
        throw s!"IR eval: const {id}: flat length mismatch: have {c.n}, expected {Spec.Shape.size s}"

/--
Evaluate a `linear` node from the external payload.

The final input axis must match `inDim`; every leading axis is preserved while the same affine map
$y=Wx+b$ is applied independently.
-/
def evalLinear {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat) (x : Spec.SomeTensor α) (outShape : Shape) :
    Except String (Spec.SomeTensor α) := do
  match payload.linear? id with
  | none => throw s!"IR eval: missing linear payload for node {id}"
  | some p =>
      let dims := x.shape.toList
      let leading : Shape := Shape.ofList dims.dropLast
      let expectedIn : Shape := leading.concat [p.inDim]
      let expectedOut : Shape := leading.concat [p.outDim]
      let xT ← expectShape (α := α) (expected := expectedIn) x
      if hOut : outShape = expectedOut then
        let y := linearLeading leading p.W p.b xT
        pure (Spec.SomeTensor.mk (α := α) outShape (hOut ▸ y))
      else
        throw <|
          s!"IR eval: linear {id}: declared outShape mismatch: {repr outShape} vs " ++
            s!"expected {repr expectedOut}"

/-- Evaluate arbitrary-rank max pooling over the spatial suffix of a shape-erased tensor. -/
def evalMaxPool {α : Type} [Context α] [DecidableEq Shape]
    (config : WindowConfig) (x : Spec.SomeTensor α) :
    Except String (Spec.SomeTensor α) := do
  let plan ← OpContracts.planPool "max_pool" config x.shape
  let input : Tensor α (plan.leading.concat (Shape.ofList plan.spatial.toList)) :=
    plan.concat_eq.symm ▸ x.tensor
  let layer : Spec.MaxPoolSpec config.spatialRank config.kernel config.stride config.padding
      plan.kernelNonzero plan.strideNonzero := {}
  let output := Tensor.mapEach plan.leading
    (Spec.maxPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
  pure (Spec.SomeTensor.ofTensor output)

/-- Evaluate arbitrary-rank average pooling over the spatial suffix of a shape-erased tensor. -/
def evalAvgPool {α : Type} [Context α] [DecidableEq Shape]
    (config : WindowConfig) (x : Spec.SomeTensor α) :
    Except String (Spec.SomeTensor α) := do
  let plan ← OpContracts.planPool "avg_pool" config x.shape
  let input : Tensor α (plan.leading.concat (Shape.ofList plan.spatial.toList)) :=
    plan.concat_eq.symm ▸ x.tensor
  let layer : Spec.AvgPoolSpec config.spatialRank config.kernel config.stride config.padding
      plan.kernelNonzero plan.strideNonzero := {}
  let output := Tensor.mapEach plan.leading
    (Spec.avgPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
  pure (Spec.SomeTensor.ofTensor output)

/-- Evaluate an arbitrary-rank convolution independently over every leading index. -/
def evalConv {α : Type} [Context α]
    (payload : Payload α) (id : Nat) (config : ConvConfig) (x : Spec.SomeTensor α) :
    Except String (Spec.SomeTensor α) := do
  let _ ←
    OpContracts.inferConvConfigOutShape "conv" config x.shape
  match payload.conv? id with
  | none => throw s!"IR eval: missing conv payload for node {id}"
  | some params =>
      unless params.matchesConfig config do
        throw s!"IR eval: conv {id}: payload does not match the node configuration"
      let leading : Shape := Shape.ofList (x.shape.toList.take config.channelAxis)
      let inputShape := params.inputShape leading
      match @decEq Shape inferInstance x.shape inputShape with
      | isTrue hInput =>
        let input : Tensor α inputShape := hInput ▸ x.tensor
        let output := Tensor.mapEach leading
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          input
        pure (Spec.SomeTensor.ofTensor output)
      | isFalse _ =>
        throw <|
          s!"IR eval: conv {id}: payload input shape {repr inputShape} does not match " ++
            s!"parent shape {repr x.shape}"

/-- Evaluate fixed-statistics BatchNorm along an arbitrary channel axis. -/
def evalBatchNorm {α : Type} [Context α]
    (payload : Payload α) (id channelAxis channels : Nat) (x : Spec.SomeTensor α) :
    Except String (Spec.SomeTensor α) := do
  let _ ← OpContracts.inferBatchNormEvalOutShape channelAxis channels x.shape
  match payload.batchNormEval? id with
  | none => throw s!"IR eval: missing batch_norm_eval payload for node {id}"
  | some params =>
      if params.c != channels then
        throw s!"IR eval: batch_norm_eval {id}: payload and node channel counts differ"
      let dims : List Nat := x.shape.toList
      let leading : Shape := Shape.ofList (dims.take channelAxis)
      let spatial : Shape := Shape.ofList (dims.drop (channelAxis + 1))
      let inputShape : Shape := leading.concat (.dim params.c spatial)
      match @decEq Shape inferInstance x.shape inputShape with
      | isTrue hInput =>
        let input : Tensor α inputShape := hInput ▸ x.tensor
        let output := Tensor.mapEach leading
          (fun sample => Spec.batchNormInference sample params.mean params.var params.gamma
            params.beta params.eps)
          input
        pure (Spec.SomeTensor.ofTensor output)
      | isFalse _ =>
        throw <|
          s!"IR eval: batch_norm_eval {id}: payload input shape {repr inputShape} does not match " ++
            s!"parent shape {repr x.shape}"

/-- Layer normalization without a learned affine transform ($\gamma=1$, $\beta=0$). -/
def layerNormMatrix {α : Type} [Context α]
    (seqLen embedDim : Nat) (x : Tensor α [seqLen, embedDim])
    (gamma beta : Tensor α [embedDim]) (epsilon : α) :
    Except String (Tensor α [seqLen, embedDim]) := do
  if hSeq : seqLen > 0 then
    if hEmb : embedDim > 0 then
      pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb)
        (epsilon := epsilon))
    else
      throw s!"layernorm: embedDim must be > 0 (got {embedDim})"
  else
    throw s!"layernorm: seqLen must be > 0 (got {seqLen})"

/-- Layer normalization with the historical unit affine transform and default epsilon. -/
def layerNormWithoutAffine {α : Type} [Context α]
    (seqLen embedDim : Nat) (x : Tensor α [seqLen, embedDim]) :
    Except String (Tensor α [seqLen, embedDim]) := do
  if hSeq : seqLen > 0 then
    if hEmb : embedDim > 0 then
      let gamma : Tensor α [embedDim] := Spec.fill 1 [embedDim]
      let beta : Tensor α [embedDim] := Spec.fill 0 [embedDim]
      pure (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (x := x) (gamma := gamma) (beta := beta) (h_seq_pos := hSeq) (h_embed_pos := hEmb))
    else
      throw s!"layernorm: embedDim must be > 0 (got {embedDim})"
  else
    throw s!"layernorm: seqLen must be > 0 (got {seqLen})"

/-- Affine data for a LayerNorm matrix view after validating its normalized suffix. -/
structure LayerNormAffine (α : Type) [Context α] (embedDim : Nat) where
  gamma : Tensor α [embedDim]
  beta : Tensor α [embedDim]
  epsilon : α

/--
Resolve optional LayerNorm payload data into the vector shape consumed by the matrix semantics.

An absent payload means the standard unit scale, zero bias, and default normalization epsilon.
Learned parameters are accepted only when their declared suffix is exactly the suffix normalized by
the IR node.
-/
def resolveLayerNormAffine {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id axis : Nat) (outShape : Shape) (embedDim : Nat) :
    Except String (LayerNormAffine α embedDim) := do
  match payload.layerNorm? id with
  | none =>
      pure
        { gamma := Spec.fill (α := α) 1 [embedDim]
          beta := Spec.fill (α := α) 0 [embedDim]
          epsilon := Numbers.normalizationEpsilon }
  | some params =>
      let normalizedShape : Shape := Shape.ofList (outShape.toList.drop axis)
      match @decEq Shape inferInstance params.normalizedShape normalizedShape with
      | isTrue hShape =>
          if hSize : Shape.size normalizedShape = embedDim then
            have hVectorSize : Shape.size normalizedShape = Shape.size [embedDim] := by
              simpa [Shape.size] using hSize
            pure
              { gamma := Tensor.reshapeSpec (hShape ▸ params.gamma) hVectorSize
                beta := Tensor.reshapeSpec (hShape ▸ params.beta) hVectorSize
                epsilon := params.eps }
          else
            throw s!"IR eval: node {id}: layernorm payload has inconsistent suffix size"
      | isFalse _ =>
          throw <|
            s!"IR eval: node {id}: layernorm payload shape {repr params.normalizedShape} " ++
              s!"does not match normalized suffix {repr normalizedShape}"

/--
Decode a dynamic concat parent as a tensor with an existential leading dimension and the requested
tail shape. This is the checked boundary shared by concat evaluation and its proofs.
-/
def expectLeadingAxisInput {α : Type} [Context α]
    (i : Nat) (rest : Shape) (value : Spec.SomeTensor α) :
    Except String (Sigma fun size => Tensor α (Shape.dim size rest)) := do
  match value with
  | ⟨Shape.dim size actualRest, tensor⟩ =>
      if hRest : actualRest = rest then
        let tensor' : Tensor α (Shape.dim size rest) := by
          simpa [hRest] using tensor
        pure ⟨size, tensor'⟩
      else
        throw <|
          s!"IR eval: node {i}: concat: tail mismatch: {repr actualRest} vs {repr rest}"
  | ⟨_, _⟩ =>
      throw s!"IR eval: node {i}: concat expects rank≥1 parents, got {repr value.shape}"

/-- Fold leading-axis concat over dynamic values that already share the same tail shape. -/
def evalConcatLeadingAxisFold {α : Type} [Context α]
    (i : Nat) (nOut : Nat) (rest : Shape) (parents : Array (Spec.SomeTensor α)) :
    Except String (Spec.SomeTensor α) := do
  let sigs ← parents.mapM (expectLeadingAxisInput (α := α) i rest)
  match sigs[0]? with
  | none =>
      throw s!"IR eval: node {i}: concat internal error"
  | some s0 =>
      let outSigma :=
        (sigs.extract 1).foldl
          (fun acc nxt =>
            match acc, nxt with
            | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
                ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                  (suffix := rest)
                  t1 t2⟩)
          s0
      match outSigma with
      | ⟨nSum, tSum⟩ =>
          if h : nSum = nOut then
            let y : Tensor α (Shape.dim nOut rest) := by
              simpa [h] using tSum
            pure (Spec.SomeTensor.mk (α := α) (Shape.dim nOut rest) y)
          else
            throw <|
              s!"IR eval: node {i}: concat out dim mismatch: declared {nOut}, " ++
                s!"computed {nSum}"

/--
Evaluate a `concat` node from already evaluated parent values.

The IR concat operation accepts any valid axis.  The tensor primitive concatenates along axis `0`,
so the evaluator implements the generic case by moving the requested axis to the front, folding
`Tensor.concatAxisSpec .scalar` over the permuted parents, and moving the result back.
-/
def evalConcat {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (axis : Nat) (parents : Array (Spec.SomeTensor α)) :
    Except String (Spec.SomeTensor α) := do
  let expected ←
    match OpContracts.inferConcatOutShape axis (parents.map (fun pv => pv.shape)) with
    | .ok s => pure s
    | .error msg => throw s!"IR eval: node {i}: {msg} ({n.summary})"
  if expected != n.outShape then
    throw <|
      s!"IR eval: node {i}: concat outShape mismatch: " ++
        s!"expected={repr expected}, declared={repr n.outShape} ({n.summary})"

  if axis = 0 then
    match n.outShape with
    | Shape.dim nOut rest =>
        evalConcatLeadingAxisFold (α := α) i nOut rest parents
    | _ =>
        throw s!"IR eval: node {i}: concat expects rank≥1 outShape, got {repr n.outShape}"
  else
  let permFront ←
    match OpContracts.permMoveAxisToFront axis n.outShape with
    | .ok perm => pure perm
    | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
  let permBack ←
    match OpContracts.inversePerm permFront with
    | .ok perm => pure perm
    | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
  let outPermShape ←
    match Spec.Shape.permute? n.outShape permFront.toList with
    | some s => pure s
    | none =>
        throw <|
          s!"IR eval: node {i}: concat: internal error (invalid permutation for " ++
            s!"outShape) ({n.summary})"
  let parentsPerm : Array (Spec.SomeTensor α) ←
    parents.mapM (fun pv => do
      match permuteSomeTensor (α := α) pv permFront with
      | .ok v => pure v
      | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})")
  match outPermShape with
  | Shape.dim nOut rest =>
      let toSigma (pv : Spec.SomeTensor α) : Except String (Sigma fun n => Tensor α (Shape.dim n rest))
        := do
        match pv with
        | ⟨Shape.dim nP restP, t⟩ =>
            if hRest : restP = rest then
              let t' : Tensor α (Shape.dim nP rest) := by
                simpa [hRest] using t
              pure ⟨nP, t'⟩
            else
              throw <|
                s!"IR eval: node {i}: concat: permuted tail mismatch: {repr restP} vs " ++
                  s!"{repr rest}"
        | ⟨_, _⟩ =>
            throw s!"IR eval: node {i}: concat expects rank≥1 parents, got {repr pv.shape}"
      let sigs ← parentsPerm.mapM toSigma
      match sigs[0]? with
      | none =>
          throw s!"IR eval: node {i}: concat internal error"
      | some s0 =>
          let outSigma :=
            (sigs.extract 1).foldl
              (fun acc nxt =>
                match acc, nxt with
                | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
                    ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                      (suffix := rest)
                      t1 t2⟩)
              s0
          match outSigma with
          | ⟨nSum, tSum⟩ =>
              if h : nSum = nOut then
                let yPerm : Tensor α (Shape.dim nOut rest) := by
                  simpa [h] using tSum
                let outPerm : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) (Shape.dim nOut rest) yPerm
                let out0 ←
                  match permuteSomeTensor (α := α) outPerm permBack with
                  | .ok v => pure v
                  | .error msg => throw s!"IR eval: node {i}: concat: {msg} ({n.summary})"
                let y ← expectShape (α := α) (expected := n.outShape) out0
                pure (Spec.SomeTensor.mk (α := α) n.outShape y)
              else
                throw <|
                  s!"IR eval: node {i}: concat out dim mismatch: declared {nOut}, " ++
                    s!"computed {nSum}"
  | _ =>
      throw s!"IR eval: node {i}: concat expects rank≥1 outShape, got {repr n.outShape}"

/-- Normalize a node result to the node's declared shape, rejecting inconsistent implementations. -/
def normalizeNodeOutput {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (v : Spec.SomeTensor α) : Except String (Spec.SomeTensor α) :=
  if h : v.shape = n.outShape then
    pure (Spec.SomeTensor.mk (α := α) n.outShape (h ▸ v.tensor))
  else
    throw <|
      s!"IR eval: node {i}: produced shape mismatch: produced={repr v.shape}, " ++
        s!"declared={repr n.outShape} ({n.summary})"

@[simp]
theorem normalizeNodeOutput_declared {α : Type} [Context α] [DecidableEq Shape]
    (i : Nat) (n : Node) (t : Tensor α n.outShape) :
    normalizeNodeOutput (α := α) i n (Spec.SomeTensor.mk (α := α) n.outShape t) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape t) := by
  simp [normalizeNodeOutput, Pure.pure, Except.pure]

@[simp]
theorem normalizeNodeOutput_nodeShape {α : Type} [Context α] [DecidableEq Shape]
    (i id : Nat) (parents : Array Nat) (kind : OpKind) (s : Shape) (t : Tensor α s) :
    normalizeNodeOutput (α := α) i { id := id, parents := parents, kind := kind, outShape := s }
        ⟨s, t⟩ =
      .ok ⟨s, t⟩ := by
  simp [normalizeNodeOutput, Pure.pure, Except.pure]

/-- Read an already-evaluated parent, reporting the node and available prefix on failure. -/
@[simp] def getParentValue {α : Type} (vals : Array (Spec.SomeTensor α)) (i : Nat) (n : Node)
    (pid : Nat) :
    Except String (Spec.SomeTensor α) :=
  match vals[pid]? with
  | some parent => pure parent
  | none =>
      throw <|
        s!"IR eval: node {i}: parent {pid} has not been evaluated " ++
          s!"(available values: {vals.size}) ({n.summary})"

/-- Decode the sole parent of a unary node. -/
def unaryParentId (i : Nat) (n : Node) : Except String Nat :=
  match unaryParent? n.parents with
  | some parent => pure parent
  | none => throw s!"IR eval: node {i}: expected 1 parent ({n.summary})"

/-- Decode the two parents of a binary node. -/
def binaryParentIds (i : Nat) (n : Node) : Except String (Nat × Nat) :=
  match binaryParents? n.parents with
  | some parents => pure parents
  | none => throw s!"IR eval: node {i}: expected 2 parents ({n.summary})"

/--
Evaluate a known node from its already computed parent values.

Keeping operator dispatch separate from graph lookup lets local correctness proofs reduce only the
selected `OpKind` branch. The caller remains responsible for the graph's topological invariant.
-/
def evalNode
    {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α)) (i : Nat) (n : Node) :
    Except String (Spec.SomeTensor α) := do
  let getParent := getParentValue vals i n
  let v : Spec.SomeTensor α ←
    match n.kind with
    | .input =>
        let t ← expectShape (α := α) (expected := n.outShape) input
        pure (Spec.SomeTensor.mk (α := α) n.outShape t)
    | .const s =>
        let t ← evalConst (α := α) (payload := payload) (id := n.id) (s := s)
        pure (Spec.SomeTensor.mk (α := α) s t)
    | .permute perm => do
        let pId ← unaryParentId i n
        let vOut ← permuteSomeTensor (α := α) (v := ← getParent pId) perm
        if h : vOut.shape = n.outShape then
          pure (Spec.SomeTensor.mk (α := α) n.outShape (h ▸ vOut.tensor))
        else
          throw <|
            s!"IR eval: node {i}: permute outShape mismatch: " ++
              s!"computed={repr vOut.shape}, declared={repr n.outShape} ({n.summary})"
    | .detach => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape p)
    | .randUniform seed => do
        unless n.parents.isEmpty do
          throw s!"IR eval: node {i}: rand_uniform expects 0 parents ({n.summary})"
        let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
        let t : Tensor α n.outShape :=
          Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := n.outShape)
        pure (Spec.SomeTensor.mk (α := α) n.outShape t)
    | .bernoulliMask seed => do
        let pId ← unaryParentId i n
        let pV ← getParent pId
        match pV with
        | ⟨.scalar, Tensor.scalar keepProb⟩ =>
            let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
            let t : Tensor α n.outShape :=
              Runtime.Autograd.TorchLean.Random.mask (α := α) key keepProb (s := n.outShape)
            pure (Spec.SomeTensor.mk (α := α) n.outShape t)
        | ⟨_, _⟩ =>
            throw
              s!"IR eval: node {i}: bernoulli_mask expects scalar keepProb parent ({n.summary})"
    | .add => do
        let (aId, bId) ← binaryParentIds i n
        let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
        let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.addSpec (α := α) a b))
    | .sub => do
        let (aId, bId) ← binaryParentIds i n
        let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
        let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.subSpec (α := α) a b))
    | .mul_elem => do
        let (aId, bId) ← binaryParentIds i n
        let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
        let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.mulSpec (α := α) a b))
    | .abs => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.absSpec (α := α) p))
    | .sqrt => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.sqrtSpec (α := α) p))
    | .maxElem => do
        let (aId, bId) ← binaryParentIds i n
        let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
        let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.maxSpec (α := α) a b))
    | .minElem => do
        let (aId, bId) ← binaryParentIds i n
        let a ← expectShape (α := α) (expected := n.outShape) (← getParent aId)
        let b ← expectShape (α := α) (expected := n.outShape) (← getParent bId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.minSpec (α := α) a b))
    | .maxPool config => do
        let pId ← unaryParentId i n
        evalMaxPool (α := α) config (← getParent pId)
    | .avgPool config => do
        let pId ← unaryParentId i n
        evalAvgPool (α := α) config (← getParent pId)
    | .broadcastTo s₁ s₂ => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := s₁) (← getParent pId)
        match OpContracts.mkCanBroadcastTo? s₁ s₂ with
        | none => throw s!"IR eval: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
        | some cb =>
            let y := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb p
            pure (Spec.SomeTensor.mk (α := α) s₂ y)
    | .reduceSum axis => do
        let pId ← unaryParentId i n
        let pV ← getParent pId
        let s := pV.shape
        let pT : Tensor α s := pV.tensor
        match Spec.Shape.nonemptyAxis? (axis := axis) s with
        | none =>
            let msg :=
              s!"IR eval: node {i}: reduce_sum invalid axis={axis}" ++
                s!" for shape {repr s}"
            throw msg
        | some hAxis =>
            let hRed := hAxis.down
            let y := Tensor.reduceSum (α := α) (s := s) axis pT hRed
            pure (Spec.SomeTensor.mk (α := α) (shapeAfterSum s axis) y)
    | .reduceMean axis => do
        let pId ← unaryParentId i n
        let pV ← getParent pId
        let s := pV.shape
        let pT : Tensor α s := pV.tensor
        match Spec.Shape.nonemptyAxis? (axis := axis) s with
        | none =>
            let msg :=
              s!"IR eval: node {i}: reduce_mean invalid axis={axis}" ++
                s!" for shape {repr s}"
            throw msg
        | some hAxis =>
            let hRed := hAxis.down
            let y := Tensor.reduceMean (α := α) (s := s) axis pT hRed
            pure (Spec.SomeTensor.mk (α := α) (shapeAfterSum s axis) y)
    | .sum => do
        let pId ← unaryParentId i n
        let p ← getParent pId
        let s := p.shape
        let t : Tensor α s := p.tensor
        let v : α := Tensor.sumSpec (α := α) t
        pure (Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar v))
    | .matmul => do
        let (aId, bId) ← binaryParentIds i n
        let aV ← getParent aId
        let bV ← getParent bId
        match aV.shape.toList.reverse, bV.shape.toList.reverse with
        | inner :: rows :: leadingRev, cols :: inner' :: leadingRev' =>
            if _hLeading : leadingRev = leadingRev' then
              if _hInner : inner = inner' then
                let leading := Shape.ofList leadingRev.reverse
                let aT ← expectShape (α := α) (expected := leading.concat [rows, inner]) aV
                let bT ← expectShape (α := α) (expected := leading.concat [inner, cols]) bV
                let y : Tensor α (leading.concat [rows, cols]) := matmulLeading leading aT bT
                pure (Spec.SomeTensor.ofTensor y : Spec.SomeTensor α)
              else
                throw s!"IR eval: node {i}: matmul inner dims mismatch: {inner} vs {inner'}"
            else
              throw <| s!"IR eval: node {i}: matmul leading dimensions mismatch: " ++
                s!"{repr aV.shape} vs {repr bV.shape}"
        | _, _ =>
            throw <| s!"IR eval: node {i}: matmul expects rank≥2 inputs, got " ++
              s!"{repr aV.shape} and {repr bV.shape}"
    | .linear => do
        let pId ← unaryParentId i n
        evalLinear (α := α) (payload := payload) (id := n.id) (x := ← getParent pId) (outShape :=
          n.outShape)
    | .conv config => do
        let pId ← unaryParentId i n
        let y ← evalConv (α := α) payload n.id config (← getParent pId)
        if y.shape != n.outShape then
          throw <|
            s!"IR eval: node {i}: conv outShape mismatch: computed={repr y.shape}, " ++
              s!"declared={repr n.outShape}"
        pure y
    | .batchNormEval channelAxis channels => do
        let pId ← unaryParentId i n
        let y ← evalBatchNorm (α := α) payload n.id channelAxis channels (← getParent pId)
        if y.shape != n.outShape then
          throw <|
            s!"IR eval: node {i}: batch_norm_eval outShape mismatch: " ++
              s!"computed={repr y.shape}, declared={repr n.outShape}"
        pure y
    | .minMax channelAxis => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape
          (Activation.minMaxAxisSpec (α := α) channelAxis p))
    | .relu => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Activation.reluSpec (α := α) p))
    | .tanh => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Activation.tanhSpec (α := α) p))
    | .sin => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.sin x) p))
    | .cos => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.mapSpec (fun x => MathFunctions.cos x) p))
    | .sigmoid => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Activation.sigmoidSpec (α := α) p))
    | .exp => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.expSpec (α := α) p))
    | .log => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        -- Domain discipline: raw `log` is undefined on nonpositive inputs. The evaluator
        -- rejects that case explicitly; use `safeLogSpec`/`safeLogOp` in models that require
        -- epsilon protection.
        if Tensor.allSpec (α := α) (s := n.outShape) (fun v => decide (0 < v)) p then
          pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.logSpec (α := α) p))
        else
          throw
            "IR eval: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
    | .inv => do
        let pId ← unaryParentId i n
        let p ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        pure (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.invSpec (α := α) p))
    | .softmax axis => do
        let pId ← unaryParentId i n
        match Spec.Shape.axisInBounds? axis n.outShape with
        | none =>
            throw <| s!"IR eval: node {i}: softmax: invalid axis {axis} for rank " ++
              s!"{Spec.Shape.rank n.outShape} ({n.summary})"
        | some h =>
            expectShape (α := α) (expected := n.outShape) (← getParent pId) >>= fun p =>
              pure <| Spec.SomeTensor.mk (α := α) n.outShape <|
                @Activation.softmaxSpec α _ n.outShape axis h.down p
    | .hardMaskedSoftmax mask => do
        let pId ← unaryParentId i n
        let scores ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        let allowed ←
          match NN.IR.HardMask.toTensorAs? mask n.outShape with
          | .ok value => pure value
          | .error msg =>
              throw s!"IR eval: node {i}: hard_masked_softmax: {msg} ({n.summary})"
        pure <| Spec.SomeTensor.mk (α := α) n.outShape <|
          Spec.hardMaskedSoftmaxSpec scores allowed
    | .layernorm axis => do
        let pId ← unaryParentId i n
        let x ← expectShape (α := α) (expected := n.outShape) (← getParent pId)
        let (seqLen, embedDim) ←
          match OpContracts.layerNormMatrixDims axis n.outShape with
          | .ok p => pure p
          | .error msg => throw s!"IR eval: node {i}: layernorm: {msg} ({n.summary})"
        let view2d : Shape := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
        if hNumel : Spec.Shape.size n.outShape = Spec.Shape.size view2d then
          let x2d : Tensor α view2d :=
            Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2d) x hNumel
          let affine ← resolveLayerNormAffine payload n.id axis n.outShape embedDim
          let y2d ←
            layerNormMatrix seqLen embedDim x2d affine.gamma affine.beta affine.epsilon
          let y : Tensor α n.outShape :=
            Tensor.reshapeSpec (α := α) (s₁ := view2d) (s₂ := n.outShape) y2d hNumel.symm
          pure (Spec.SomeTensor.mk (α := α) n.outShape y)
        else
          throw <|
            s!"IR eval: node {i}: layernorm internal error: bad reshape sizes " ++
              s!"({Spec.Shape.size n.outShape} vs {Spec.Shape.size view2d}) ({n.summary})"
    | .reshape inS outS => do
        let pId ← unaryParentId i n
        let pV ← getParent pId
        let pT ← expectShape (α := α) (expected := inS) pV
        if h : Spec.Shape.size inS = Spec.Shape.size outS then
          let y := Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) pT h
          pure (Spec.SomeTensor.mk (α := α) outS y)
        else
          throw
            s!"IR eval: node {i}: reshape numel mismatch: {Spec.Shape.size inS} vs {Spec.Shape.size outS}"
    | .flatten s => do
        let pId ← unaryParentId i n
        let pV ← getParent pId
        let pT ← expectShape (α := α) (expected := s) pV
        let y := Tensor.flattenSpec (α := α) (s := s) pT
        pure (Spec.SomeTensor.mk (α := α) (.dim (Spec.Shape.size s) .scalar) y)
    | .concat axis => do
        let parents ← n.parents.mapM getParent
        evalConcat (α := α) i n axis parents
    | .transpose axis₁ axis₂ => do
        let pId ← unaryParentId i n
        let parent ← getParent pId
        let perm ← OpContracts.transposePerm parent.shape.rank axis₁ axis₂
        let output ← permuteSomeTensor (α := α) parent perm
        let tensor ← expectShape (α := α) (expected := n.outShape) output
        pure (Spec.SomeTensor.mk (α := α) n.outShape tensor)
    | .mseLoss => do
        let (yId, tId) ← binaryParentIds i n
        mseLossSomeTensor (α := α) i (← getParent yId) (← getParent tId)
  normalizeNodeOutput (α := α) i n v

/--
Evaluate node `i` after checking the graph's id discipline and retrieving the corresponding node.

`denoteAll` checks the full graph structure before repeatedly calling this one-step evaluator.
-/
def evalAt
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α)) (i : Nat) :
    Except String (Spec.SomeTensor α) := do
  let n ← g.getNode i
  evalNode (α := α) payload input vals i n

/--
Evaluate nodes `i, i+1, ...` given already computed prefix values `vals`.

This is written as a structurally recursive function so it is easy to reason about in proofs
(evaluation is “a simple loop over node ids”).
-/
def denoteAllFrom
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (i : Nat) (vals : Array (Spec.SomeTensor α)) :
    Except String (Array (Spec.SomeTensor α)) := do
  if h : i < g.nodes.size then
    let v ← evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i := i)
    denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := i + 1) (vals :=
      vals.push v)
  else
    pure vals
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) h

/--
Evaluate a graph to a table of node values.

This returns an array `vals` of length `g.size` where `vals[i]` is the value of node `i`.

We do a structural well-formedness check once up front (ids/arity/topology). For lowering-produced
graphs, the boolean `Graph.wellFormed` check is a fast path; if it fails we fall back to the
exception-producing `Graph.checkWellFormed` so callers get a readable error message.

The evaluator is total in the sense that it always returns either:
- `.ok vals` (all nodes evaluated successfully), or
- `.error msg` describing the first failure (malformed IR, missing payload, or a local shape error).
-/
def denoteAll
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) : Except String (Array (Spec.SomeTensor α)) := do
  -- Fast path: lowering-produced graphs typically satisfy the boolean `wellFormed` discipline.
  if g.wellFormed then
    pure ()
  else
    g.checkWellFormed
  denoteAllFrom (α := α) (g := g) (payload := payload) (input := input) (i := 0)
    (vals := #[])

/-! ## Scoped notation -/

/--
Scoped notation for evaluating a graph to all node values.

Use with:

```lean
open scoped IR
g⟦payload, input⟧
```
-/
scoped[IR] notation g "⟦" payload ", " input "⟧" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- ASCII alternative to `g⟦payload, input⟧`. -/
scoped[IR] notation g "[[" payload ", " input "]]" =>
  _root_.NN.IR.Graph.denoteAll g payload input

/-- Evaluate the graph and return the value at `outputId`. -/
def denote
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (outputId : Nat) : Except String (Spec.SomeTensor α) :=
      do
  let vals ← denoteAll (α := α) (g := g) (payload := payload) (input := input)
  match vals[outputId]? with
  | none => throw s!"IR eval: outputId out of bounds: {outputId}"
  | some v => pure v

end Graph

end NN.IR
