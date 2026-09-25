/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core
public import NN.IR.Payload
public import NN.IR.Semantics
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor.SomeTensor

/-!
Shared definitions for the graph CROWN engine.

This file contains the rank-one tensor representation, parameter stores, interval boxes, shape
permutation helpers, and tensor casts used by the IBP, derivative, affine, CROWN, and backward
objective passes.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/--
An existentially sized rank-one tensor used by the flat LiRPA engine.

Graph nodes carry dimensions discovered while lowering, so the dimension cannot always appear in
the surrounding static type. The field `n` is the hidden tensor dimension, not duplicate runtime
storage.
-/
structure FlatTensor (α : Type) [Context α] where
  /-- Number of scalar entries. -/
  n : Nat
  /-- Rank-one tensor payload (shape `.dim n .scalar`). -/
  v : Tensor α [n]

-- The flat-tensor engine is the canonical executable path for the current graph verifier.

/--
Parameters for a linear layer `y = W*x + b` in flattened form.

`m` is the output dimension and `n` is the input dimension.
-/
structure LinParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α [m, n]
  /-- Bias tensor `b` (shape `m`). -/
  b : Tensor α [m]

/-- Matrix parameters for bias-free matmul: y = W x. -/
structure MatParams (α : Type) [Context α] where
  /-- Output dimension. -/
  m : Nat
  /-- Input dimension. -/
  n : Nat
  /-- Weight matrix `W` (shape `m × n`). -/
  w : Tensor α [m, n]

/-- Coordinate on `axis` corresponding to a row-major flat index. -/
def axisCoordinateOfFlat (shape : Shape) (axis idx : Nat) : Nat :=
  let dims := shape.toList
  let axisStride := (dims.drop (axis + 1)).prod
  if axisStride = 0 then 0 else idx / axisStride

/-- Eval BatchNorm scale for one channel. -/
def batchNormEvalScale (cfg : NN.IR.BatchNormEvalParams α) (ci : Fin cfg.c) : α :=
  match get cfg.gamma ci, get cfg.var ci with
  | .scalar gamma, .scalar var =>
      gamma / MathFunctions.sqrt (max var Numbers.zero + cfg.eps)

/-- Eval-mode BatchNorm bias after folding one channel's running statistics into an affine map. -/
def batchNormEvalBias (cfg : NN.IR.BatchNormEvalParams α) (ci : Fin cfg.c) : α :=
  match get cfg.beta ci, get cfg.mean ci with
  | .scalar beta, .scalar mean =>
      beta - mean * batchNormEvalScale (α := α) cfg ci

/--
Build the exact diagonal affine form for eval-mode BatchNorm on an arbitrary channel axis.

The graph records the channel axis while the payload stores one scale and bias per channel. A
malformed axis or mismatched channel extent has no verifier transfer rule.
-/
def batchNormEvalLinear? (parentShape : Shape) (channelAxis : Nat)
    (cfg : NN.IR.BatchNormEvalParams α) : Option (LinParams α) := do
  let channels ← parentShape.toList[channelAxis]?
  if hcfg : cfg.c = 0 then
    none
  else if channels = cfg.c then
    haveI : NeZero cfg.c := ⟨hcfg⟩
    let outDim := parentShape.size
    let weight : Tensor α [outDim, outDim] :=
      Tensor.dim (fun oi =>
        Tensor.dim (fun ii =>
          let ch := axisCoordinateOfFlat parentShape channelAxis oi.val
          let scale := batchNormEvalScale (α := α) cfg (Fin.ofNat cfg.c ch)
          Tensor.scalar (if decide (oi.val = ii.val) then scale else Numbers.zero)))
    let bias : Tensor α [outDim] :=
      Tensor.dim (fun oi =>
        let ch := axisCoordinateOfFlat parentShape channelAxis oi.val
        Tensor.scalar (batchNormEvalBias (α := α) cfg (Fin.ofNat cfg.c ch)))
    some { m := outDim, n := outDim, w := weight, b := bias }
  else
    none

/--
Parameters keyed by node id (weights, biases, constants, and seeded input boxes).

This is kept compact: it is the graph interpreter used to run IBP/CROWN on a pure `Graph`
without pulling in a heavyweight runtime.
-/
structure ParamStore (α : Type) [Context α] where
  /-- Seed boxes for designated input nodes (`id -> FlatBox`). -/
  inputBoxes : Std.HashMap Nat (FlatBox α) := Std.HashMap.emptyWithCapacity
  /-- Constants (`id -> FlatTensor`). -/
  constVals  : Std.HashMap Nat (FlatTensor α) := Std.HashMap.emptyWithCapacity
  /-- Linear layer params (`id -> (W,b)`). -/
  linearWB   : Std.HashMap Nat (LinParams α) := Std.HashMap.emptyWithCapacity
  /-- Matmul params (`id -> W`) for bias-free multiplication. -/
  matmulW    : Std.HashMap Nat (MatParams α) := Std.HashMap.emptyWithCapacity
  /-- Convolution specs (`id -> convolution configuration`). -/
  convCfg : Std.HashMap Nat (NN.IR.ConvParams α) := Std.HashMap.emptyWithCapacity
  /-- Eval-mode BatchNorm parameters (`id -> gamma/beta/running stats`). -/
  batchNormEval : Std.HashMap Nat (NN.IR.BatchNormEvalParams α) :=
    Std.HashMap.emptyWithCapacity
  /-- Affine LayerNorm parameters keyed by node id.

  The current CROWN transfer rule supports only the pure normalization recorded by an absent
  payload. Retaining an explicit map lets the verifier reject affine LayerNorm rather than silently
  replacing its scale, bias, or epsilon with defaults. -/
  layerNorm : Std.HashMap Nat (NN.IR.LayerNormParams α) := Std.HashMap.emptyWithCapacity

namespace ParamStore

/-- Insert an input interval box for a graph node. -/
def seedInputBox {α : Type} [Context α]
    (ps : ParamStore α) (inputId : Nat) (xB : FlatBox α) : ParamStore α :=
  { ps with inputBoxes := ps.inputBoxes.insert inputId xB }

/-- Seed a graph input with a uniform `ℓ∞` box around a shaped tensor. -/
def seedLInfBall {α : Type} [Context α] {s : Shape}
    (ps : ParamStore α) (inputId : Nat) (center : Tensor α s) (eps : α) : ParamStore α :=
  ps.seedInputBox inputId <| FlatBox.lInfBall (α := α) center eps

end ParamStore

/--
Whether the current dense convolution transfer exactly matches the corresponding IR semantics.

The flattened CROWN rule implements channel-first convolution without leading batch axes, channel
groups, dilation, or asymmetric padding. The payload must also agree with the configuration stored
in the graph node; otherwise executable IR evaluation and bound propagation would denote different
operators.
-/
def convTransferSupported (config : NN.IR.ConvConfig) (cfg : NN.IR.ConvParams α)
    (parentShape outShape : Shape) : Bool :=
  cfg.matchesConfig config &&
    config.channelAxis == 0 &&
    config.groups == 1 &&
    config.dilation.toList == List.replicate config.spatialRank 1 &&
    config.paddingAfter.toList == config.padding.toList &&
    parentShape == cfg.inputShape .scalar &&
    outShape == cfg.outputShape .scalar

/--
Check the graph-level semantic restrictions imposed by the current CROWN engine.

Unsupported convolutions, non-leading-axis concatenation, and payload-bearing LayerNorm nodes are
left without bounds. This predicate is shared by forward, backward, and certificate replay paths so
one path cannot silently reinterpret a node rejected by another.
-/
def crownNodeSemanticsSupported (nodes : Array Node) (ps : ParamStore α) (id : Nat) : Bool :=
  match nodes[id]? with
  | none => false
  | some node =>
      match node.kind with
      | .conv config =>
          match node.parents with
          | #[parentId] =>
              match nodes[parentId]?, ps.convCfg[id]? with
              | some parent, some cfg =>
                  convTransferSupported (α := α) config cfg parent.outShape node.outShape
              | _, _ => false
          | _ => false
      | .concat axis =>
          if axis != 0 then
            false
          else
            match node.parents with
            | #[leftId, rightId] =>
                match nodes[leftId]?, nodes[rightId]? with
                | some left, some right =>
                    match OpContracts.inferConcatOutShape axis #[left.outShape, right.outShape] with
                    | .ok expected => expected == node.outShape
                    | .error _ => false
                | _, _ => false
            | _ => false
      | .layernorm axis =>
          match node.parents, ps.layerNorm[id]? with
          | #[parentId], none =>
              match nodes[parentId]? with
              | some parent =>
                  axis == node.outShape.rank - 1 && parent.outShape == node.outShape
              | none => false
          | _, _ => false
      | _ => true

/-- Whether every node in a graph is interpreted exactly by the current CROWN engine. -/
def crownGraphSemanticsSupported (g : Graph) (ps : ParamStore α) : Bool :=
  (List.range g.nodes.size).all (crownNodeSemanticsSupported (α := α) g.nodes ps)

/-- Read a node's interval box from an IBP-style result array. -/
def outputBox? {α : Type} [Context α]
    (boxes : Array (Option (FlatBox α))) (outId : Nat) : Except String (FlatBox α) := do
  match boxes[outId]? with
  | some (some outB) => pure outB
  | some none => throw s!"output box missing at node {outId}"
  | none => throw s!"output node {outId} is out of bounds for {boxes.size} boxes"

/-- Default inhabitant for `FlatBox` (a 0-dimensional box at `0`). -/
instance : Inhabited (FlatBox α) where
  default := { dim := 0, lo := Spec.fill (α:=α) 0 (.dim 0 .scalar), hi := Spec.fill (α:=α) 0 (.dim 0
    .scalar) }

/-- Elementwise product of two FlatBoxes (interval product per component). Requires equal dims. -/
@[expose] public def boxMulElem (B1 B2 : FlatBox α) : Option (FlatBox α) :=
  match B1, B2 with
  | ⟨n1, l1, u1⟩, ⟨n2, l2, u2⟩ =>
    if h : n1 = n2 then
      by
        cases h
        let lo :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulDown lx ly; let p2 := BoundOps.mulDown lx uy
                let p3 := BoundOps.mulDown ux ly; let p4 := BoundOps.mulDown ux uy
                let m1 := min2 p1 p2
                let m2 := min2 p3 p4
                Tensor.scalar (min2 m1 m2))
        let hi :=
          match l1, u1, l2, u2 with
          | .dim l1, .dim u1, .dim l2, .dim u2 =>
            Tensor.dim (fun i =>
              match l1 i, u1 i, l2 i, u2 i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
                let p1 := BoundOps.mulUp lx ly; let p2 := BoundOps.mulUp lx uy
                let p3 := BoundOps.mulUp ux ly; let p4 := BoundOps.mulUp ux uy
                let m1 := max2 p1 p2
                let m2 := max2 p3 p4
                Tensor.scalar (max2 m1 m2))
        exact some { dim := n1, lo := lo, hi := hi }
    else none

/-- Chain-rule multiplication for derivative intervals. Returns `none` on dimension mismatch. -/
def chainMul (dZ dF : FlatBox α) : Option (FlatBox α) :=
  boxMulElem (α:=α) dZ dF

/-- Convert a dependent `Box` of shape `.dim n .scalar` into a `FlatBox` with `dim := n`. -/
@[expose]
def toFlatBox (n : Nat) (B : Box α (.dim n .scalar)) : FlatBox α :=
  { dim := n, lo := B.lo, hi := B.hi }

/-- Convert a `FlatBox` to a dependent `Box` at shape `.dim B.dim .scalar`. -/
@[expose]
public def ofFlatBox (B : FlatBox α) : Box α (.dim B.dim .scalar) :=
  { lo := B.lo, hi := B.hi }

/-- Add two flat interval boxes coordinatewise; dimension mismatches preserve the left box. -/
@[expose]
public def boxAdd (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact
            { dim := n1
              lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
              hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Interval subtraction on `FlatBox` endpoints (sound enclosure). -/
@[expose]
public def boxSub (B1 B2 : FlatBox α) : FlatBox α :=
  match B1 with
  | ⟨n1, lo1, hi1⟩ =>
    match B2 with
    | ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          -- Sound interval subtraction: [l1,u1] - [l2,u2] = [l1 - u2, u1 - l2]
          exact
            { dim := n1
              lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
              hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Directed lower and upper sums of all coordinates in a flat box. -/
private def boxSumEndpoints (B : FlatBox α) : α × α :=
  let loValues := match B.lo with | .dim values => values
  let hiValues := match B.hi with | .dim values => values
  let lo := (List.finRange B.dim).foldl (fun acc i =>
    match loValues i with
    | .scalar x => BoundOps.addDown acc x) Numbers.zero
  let hi := (List.finRange B.dim).foldl (fun acc i =>
    match hiValues i with
    | .scalar x => BoundOps.addUp acc x) Numbers.zero
  (lo, hi)

/-- Sum all coordinates of a flat box with directed accumulation. -/
def boxSum (B : FlatBox α) : FlatBox α :=
  let (lo, hi) := boxSumEndpoints (α := α) B
  { dim := 1
    lo := Spec.fill (α := α) lo (.dim 1 .scalar)
    hi := Spec.fill (α := α) hi (.dim 1 .scalar) }

/-- Average all coordinates of a nonempty flat box with directed division. -/
def boxMean? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) := do
  if B.dim = 0 then
    none
  else
    let (lo, hi) := boxSumEndpoints (α := α) B
    let n : α := B.dim
    let (meanLo, meanHi) ← NonlinearBoundOps.divBounds lo hi n n
    pure
      { dim := 1
        lo := Spec.fill (α := α) meanLo (.dim 1 .scalar)
        hi := Spec.fill (α := α) meanHi (.dim 1 .scalar) }

/-- Apply ReLU to both endpoints of a `FlatBox` (monotone activation, so endpoints suffice). -/
@[expose]
public def boxRelu (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.lo
    hi := Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := α) x) B.hi }

/-! ### MinMax

`OpKind.minMax` pairs adjacent entries along a channel axis and replaces each
pair by its minimum and its maximum.  The engines here work on flattened
vectors, but nothing is lost: the node carries both its `outShape` and its
`channelAxis`, so the pairing is recovered as a stride and an extent.

`minMaxFlatStride` is the product of the extents *after* the channel axis, and
`minMaxFlatExtent` the extent *at* it.  A flat index `i` then sits at channel
`(i / stride) % extent`, and pairs with `i ± stride`.
-/

/-- The extent of `s` at `axis`, or `0` if the axis is out of range. -/
public def minMaxFlatExtent (axis : Nat) (s : Shape) : Nat :=
  (Spec.Shape.toList s)[axis]?.getD 0

/-- The flat stride of `axis` in `s`: the product of the extents after it. -/
public def minMaxFlatStride (axis : Nat) (s : Shape) : Nat :=
  ((Spec.Shape.toList s).drop (axis + 1)).foldl (· * ·) 1

/-- The flat index paired with `i`.  Degenerate parameters make it its own
partner, which leaves the corresponding entry unchanged. -/
public def minMaxFlatPartner (stride extent i : Nat) : Nat :=
  if stride = 0 || extent = 0 then i
  else if ((i / stride) % extent) % 2 = 0 then i + stride else i - stride

/-- Whether flat index `i` receives the *minimum* of its pair. -/
public def minMaxFlatTakesMin (stride extent i : Nat) : Bool :=
  stride != 0 && extent != 0 && ((i / stride) % extent) % 2 == 0

/-- Interval bounds for MinMax.  Sound because `min` and `max` are monotone in
both arguments, so the extremes of `min a b` over a pair of boxes are `min` of
the two lower bounds and `min` of the two upper bounds, and likewise for `max`.

Exact, not merely sound: every corner is attained. -/
public def boxMinMax (stride extent : Nat) (B : FlatBox α) : FlatBox α :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
      let partner (i : Fin B.dim) : Fin B.dim :=
        let j := minMaxFlatPartner stride extent i.val
        if h : j < B.dim then ⟨j, h⟩ else i
      let combine (v : Fin B.dim → Tensor α .scalar) (i : Fin B.dim) : Tensor α .scalar :=
        match v i, v (partner i) with
        | .scalar a, .scalar b =>
            Tensor.scalar <|
              if minMaxFlatTakesMin stride extent i.val then
                (if a < b then a else b)
              else
                (if a < b then b else a)
      { dim := B.dim
        lo := Tensor.dim (combine lo)
        hi := Tensor.dim (combine hi) }

/-- The pairwise interval **hull** under the MinMax pairing: entry `i` becomes the
hull of `i` and its partner.

This is what a *derivative* passes through MinMax.  Unlike every other
activation here, MinMax's Jacobian is not diagonal: the output at `i` is one of
the two inputs of its pair, so a directional derivative arriving at `i` may have
come from either.  Taking the hull of the pair therefore encloses both
selections, whichever the actual comparison makes — and does so without needing
to know which, so it is sound at ties too. -/
public def boxPairHull (stride extent : Nat) (B : FlatBox α) : FlatBox α :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
      let partner (i : Fin B.dim) : Fin B.dim :=
        let j := minMaxFlatPartner stride extent i.val
        if h : j < B.dim then ⟨j, h⟩ else i
      { dim := B.dim
        lo := Tensor.dim fun i =>
          match lo i, lo (partner i) with
          | .scalar a, .scalar b => Tensor.scalar (if a < b then a else b)
        hi := Tensor.dim fun i =>
          match hi i, hi (partner i) with
          | .scalar a, .scalar b => Tensor.scalar (if a < b then b else a) }

/-- Componentwise absolute value bounds. Soundly encloses `abs` over each interval component. -/
def boxAbs (B : FlatBox α) : FlatBox α :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
      let lo' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let minAbs :=
                if l < Numbers.zero then
                  if Numbers.zero < u then Numbers.zero else (if al < au then al else au)
                else
                  if al < au then al else au
              Tensor.scalar minAbs)
      let hi' :=
        Tensor.dim (fun i =>
          match lo i, hi i with
          | .scalar l, .scalar u =>
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              let maxAbs := if al > au then al else au
              Tensor.scalar maxAbs)
      { dim := B.dim, lo := lo', hi := hi' }

namespace boxUnaryEnclosure

/-- Traverse a finite family without converting its index to an untyped list. -/
private def traverseFin {β : Type} {n : Nat} (f : Fin n → Option β) : Option (Fin n → β) :=
  if h : ∀ i, (f i).isSome then
    some fun i => (f i).get (h i)
  else
    none

private theorem traverseFin_eq_some_iff {β : Type} {n : Nat}
    {f : Fin n → Option β} {g : Fin n → β} :
    traverseFin f = some g ↔ ∀ i, f i = some (g i) := by
  unfold traverseFin
  split_ifs with h
  · constructor
    · intro hfg i
      have : (fun i => (f i).get (h i)) = g := Option.some.inj hfg
      rw [← this]
      exact (Option.some_get (h i)).symm
    · intro hfg
      congr
      funext i
      obtain ⟨_, hi⟩ := Option.eq_some_iff_get_eq.mp (hfg i)
      exact hi
  · constructor
    · simp
    · intro hfg
      exfalso
      apply h
      intro i
      simp [hfg i]

end boxUnaryEnclosure

/-- Apply a scalar interval enclosure coordinatewise to a flat box. -/
def boxUnaryEnclosure? [NonlinearBoundOps α]
    (enclose : α → α → Option (α × α)) (B : FlatBox α) : Option (FlatBox α) := do
  let lo := match B.lo with | .dim values => values
  let hi := match B.hi with | .dim values => values
  let bounds ← boxUnaryEnclosure.traverseFin fun i =>
    match lo i, hi i with
    | .scalar l, .scalar u => enclose l u
  let lower : Tensor α [B.dim] :=
    Tensor.dim fun i => Tensor.scalar (bounds i).1
  let upper : Tensor α [B.dim] :=
    Tensor.dim fun i => Tensor.scalar (bounds i).2
  pure { dim := B.dim, lo := lower, hi := upper }

/--
The real value represented by each coordinate of `x` lies between the interpreted endpoints of
`B`. This is the semantic relation used to connect executable endpoint arithmetic to the real graph
semantics.
-/
def EnclosesReal [LawfulBoundOps α]
    (B : FlatBox α) (x : Tensor ℝ [B.dim]) : Prop :=
  ∀ i,
    LawfulBoundOps.toReal (Spec.Tensor.getScalar B.lo i) ≤ Spec.Tensor.getScalar x i ∧
      Spec.Tensor.getScalar x i ≤ LawfulBoundOps.toReal (Spec.Tensor.getScalar B.hi i)

/-- Dimension-aware enclosure of a real vector by a backend box. -/
def EnclosesRealValue [LawfulBoundOps α] {n : Nat}
    (B : FlatBox α) (x : Tensor ℝ [n]) : Prop :=
  ∃ h : B.dim = n, EnclosesReal B (h.symm ▸ x)

/-- A lawful scalar transfer remains sound when applied coordinatewise to a flat graph box. -/
theorem boxUnaryEnclosure?_enclosesReal [LawfulBoundOps α] [NonlinearBoundOps α]
    (f : ℝ → ℝ) (enclose : α → α → Option (α × α))
    (henclose : UnaryEnclosure (α := α) f enclose) (B : FlatBox α)
    (x : Tensor ℝ [B.dim]) (hx : EnclosesReal B x)
    {out : FlatBox α} (hout : boxUnaryEnclosure? (α := α) enclose B = some out) :
    EnclosesRealValue out (Tensor.mapSpec f x) := by
  cases hlo : B.lo with
  | dim lo =>
    cases hhi : B.hi with
    | dim hi =>
      cases hxv : x with
      | dim xv =>
        simp only [boxUnaryEnclosure?, hlo, hhi] at hout
        obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
        have hpoint := boxUnaryEnclosure.traverseFin_eq_some_iff.mp hbounds
        have houtEq :
            out =
              { dim := B.dim
                lo := Tensor.dim fun i => Tensor.scalar (bounds i).1
                hi := Tensor.dim fun i => Tensor.scalar (bounds i).2 } := by
          exact (Option.some.inj hout).symm
        subst out
        refine ⟨rfl, ?_⟩
        intro i
        cases hloi : lo i with
        | scalar l =>
          cases hhii : hi i with
          | scalar u =>
            cases hxvi : xv i with
            | scalar v =>
              have hx_i := hx i
              have htransfer : enclose l u = some (bounds i) := by
                simpa [hloi, hhii] using hpoint i
              have hscalar := henclose htransfer
                (by simpa [EnclosesReal, Spec.Tensor.getScalar, hlo, hhi, hxv, hloi, hhii, hxvi]
                  using hx_i.1)
                (by simpa [EnclosesReal, Spec.Tensor.getScalar, hlo, hhi, hxv, hloi, hhii, hxvi]
                  using hx_i.2)
              simpa [EnclosesReal, Spec.Tensor.getScalar, Tensor.mapSpec, hxv, hxvi] using hscalar

/-- Componentwise square-root enclosure supplied by the scalar backend. -/
def boxSqrt? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.sqrtBounds B

/-- Componentwise reciprocal bounds, failing when an input coordinate interval crosses zero. -/
@[expose]
def boxInv? [NonlinearBoundOps α] (B : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi => NonlinearBoundOps.divBounds Numbers.one Numbers.one lo hi) B

/-- Derivative range for `exp`; `exp' = exp`. -/
def derivBoxExp? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α) NonlinearBoundOps.expBounds zB

/-- Derivative range for `log`; `log' x = 1/x` on a strictly positive interval. -/
def derivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > Numbers.zero then
        NonlinearBoundOps.divBounds Numbers.one Numbers.one lo hi
      else
        none) zB

/-- Second-derivative range for `log`; `log'' x = -1/x²` on a positive interval. -/
def secondDerivBoxLog? [NonlinearBoundOps α] (zB : FlatBox α) : Option (FlatBox α) :=
  boxUnaryEnclosure? (α := α)
    (fun lo hi =>
      if lo > Numbers.zero then do
        let squareLo := BoundOps.mulDown lo lo
        let squareHi := BoundOps.mulUp hi hi
        let reciprocal ←
          NonlinearBoundOps.divBounds Numbers.one Numbers.one squareLo squareHi
        pure (-reciprocal.2, -reciprocal.1)
      else
        none) zB

/-- Negate an interval box by swapping and negating its endpoints. -/
def boxNeg (B : FlatBox α) : FlatBox α :=
  { dim := B.dim
    lo := Tensor.mapSpec (fun x => -x) B.hi
    hi := Tensor.mapSpec (fun x => -x) B.lo }

/-- Apply a full axis permutation to a shape-tagged tensor when the permutation is valid. -/
def permuteSomeTensor? {α : Type} [Context α]
    (v : Spec.SomeTensor α) (perm : Array Nat) : Option (Spec.SomeTensor α) :=
  (NN.IR.Graph.permuteSomeTensor v perm).toOption

/-- Convert a row-major flat index into coordinates for the given dimensions. -/
private def flatCoordinates (dims : Array Nat) (index : Nat) : Array Nat := Id.run do
  let mut coordinates := Array.replicate dims.size 0
  let mut remainder := index
  for axis in [0:dims.size] do
    let tailSize := (dims.extract (axis + 1) dims.size).foldl (· * ·) 1
    coordinates := coordinates.set! axis (remainder / tailSize)
    remainder := remainder % tailSize
  return coordinates

/-- Convert row-major coordinates back into a flat index. -/
private def coordinatesFlatIndex (dims coordinates : Array Nat) : Nat := Id.run do
  let mut index := 0
  for axis in [0:min dims.size coordinates.size] do
    index := index * dims[axis]! + coordinates[axis]!
  return index

/--
Return the flat-coordinate permutation induced by an axis permutation.

`perm` follows the tensor convention used by `Shape.permute?`: output axis `i` is read from input
axis `perm[i]`. The resulting function therefore maps each output flat coordinate to the input
flat coordinate from which its value is taken. Invalid permutations and inconsistent dimensions
are rejected.
-/
def flatAxisPermutation? (sourceShape : Shape) (perm : Array Nat) (n : Nat) :
    Option (Fin n → Fin n) := do
  let targetShape ← Spec.Shape.permute? sourceShape perm.toList
  if sourceShape.size != n || targetShape.size != n then
    none
  else if hn : n = 0 then
    none
  else
    let _ : NeZero n := ⟨hn⟩
    let inverse ← (NN.IR.OpContracts.inversePerm perm).toOption
    let sourceDims := sourceShape.toArray
    let targetDims := targetShape.toArray
    pure fun outputIndex =>
      let targetCoordinates := flatCoordinates targetDims outputIndex.val
      let sourceCoordinates := inverse.map (fun targetAxis => targetCoordinates.getD targetAxis 0)
      Fin.ofNat n (coordinatesFlatIndex sourceDims sourceCoordinates)

/-- Componentwise max bounds: `max(x,y)` over interval boxes. -/
def boxMaxElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.maxSpec (α := α) lo1 lo2
                  hi := Tensor.maxSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/-- Componentwise min bounds: `min(x,y)` over interval boxes. -/
def boxMinElem (B1 B2 : FlatBox α) : FlatBox α :=
  match B1, B2 with
  | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
      if h : n1 = n2 then
        by
          cases h
          exact { dim := n1
                  lo := Tensor.minSpec (α := α) lo1 lo2
                  hi := Tensor.minSpec (α := α) hi1 hi2 }
      else
        { dim := n1, lo := lo1, hi := hi1 }

/--
Componentwise square of an interval box: for each component `[l,u]` produce `[min (l^2,u^2), max
  (l^2,u^2)]`, with `0` as the minimum when the interval crosses `0`.

The body is exposed because the proof layer theorem module unfolds this executable rule when
proving dimension preservation and pointwise enclosure.
-/
@[expose] def boxSquare (B : FlatBox α) : FlatBox α :=
  let loF : Fin B.dim → Tensor α .scalar :=
    match B.lo with
    | .dim f => f
  let hiF : Fin B.dim → Tensor α .scalar :=
    match B.hi with
    | .dim f => f
  let lo' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let minSq :=
          if l < Numbers.zero then
            if Numbers.zero < u then Numbers.zero else (if l2 < u2 then l2 else u2)
          else (if l2 < u2 then l2 else u2)
        Tensor.scalar minSq)
  let hi' :=
    Tensor.dim (fun i =>
      match loF i, hiF i with
      | .scalar l, .scalar u =>
        let l2 := l * l
        let u2 := u * u
        let maxSq := if l2 > u2 then l2 else u2
        Tensor.scalar maxSq)
  { dim := B.dim, lo := lo', hi := hi' }

/-- Interval multiplication for scalar endpoints: given `[aLo,aHi]` and `[bLo,bHi]`, return bounds
  on the product. -/
def intervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1 := aLo * bLo
  let p2 := aLo * bHi
  let p3 := aHi * bLo
  let p4 := aHi * bHi
  let lo1 := if p1 < p2 then p1 else p2
  let lo2 := if p3 < p4 then p3 else p4
  let lo  := if lo1 < lo2 then lo1 else lo2
  let hi1 := if p1 > p2 then p1 else p2
  let hi2 := if p3 > p4 then p3 else p4
  let hi  := if hi1 > hi2 then hi1 else hi2
  (lo, hi)

/-- Length of the last axis of a shape; scalars are treated as length one. -/
def lastDimLen : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ rest => lastDimLen rest

/-- Runtime witness that one shape can broadcast to another. -/
def mkCanBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | s₁, s₂ =>
    if hlt : Spec.Shape.rank s₁ < Spec.Shape.rank s₂ then
      match s₂ with
      | .scalar => none
      | .dim n₂ t₂ =>
        (mkCanBroadcastTo? s₁ t₂).map (fun tail =>
          Shape.CanBroadcastTo.expand_dims (n := n₂) (s₁ := s₁) (s₂ := t₂) tail)
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

/-- Reinterpret a flattened tensor as shape `s` when the element counts agree. -/
def ibpUnflatten {s : Shape} (dim : Nat) (t : Tensor α [dim]) (h : dim =
  Spec.Shape.size s) :
    Tensor α s :=
  let t' : Tensor α [Spec.Shape.size s] := by
    simpa [h] using t
  Tensor.unflattenSpec (α := α) s t'

/-- IBP rule for broadcasting a flattened input box to a target shape. -/
def ibpBroadcastTo (s₁ s₂ : Shape) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s₁ then
    match mkCanBroadcastTo? s₁ s₂ with
    | none => none
    | some cb =>
        let xLo : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.lo h
        let xHi : Tensor α s₁ := ibpUnflatten (α := α) (s := s₁) Xin.dim Xin.hi h
        let yLo : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xLo
        let yHi : Tensor α s₂ := Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb xHi
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size s₂, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by summing along one axis. -/
def ibpReduceSumAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceSum (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceSum (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/-- IBP rule for reducing a shaped box by averaging along one axis. -/
def ibpReduceMeanAxis (axis : Nat) (Xin : FlatBox α) (s : Shape) : Option (FlatBox α) :=
  if h : Xin.dim = Spec.Shape.size s then
    match Spec.Shape.nonemptyAxis? (axis := axis) s with
    | none => none
    | some hAxis =>
        let hRed := hAxis.down
        let xLo : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.lo h
        let xHi : Tensor α s := ibpUnflatten (α := α) (s := s) Xin.dim Xin.hi h
        let yLo := Tensor.reduceMean (α := α) (s := s) axis xLo hRed
        let yHi := Tensor.reduceMean (α := α) (s := s) axis xHi hRed
        let outS := Tensor.shapeAfterSum s axis
        let flatLo := Tensor.flattenSpec (α := α) yLo
        let flatHi := Tensor.flattenSpec (α := α) yHi
        some { dim := Spec.Shape.size outS, lo := flatLo, hi := flatHi }
  else
    none

/--
Format-independent softmax enclosure on a flattened tensor.

A singleton row is exactly one. Every coordinate of a longer row lies in `[0,1]`. This deliberately
forgoes the tighter exponential formula above so executable checking does not assume a directed
transcendental implementation that its scalar backend has not supplied.
-/
def ibpSoftmaxRange (s : Shape) (dim : Nat) : FlatBox α :=
  let rowLength := lastDimLen s
  if rowLength = 1 then
    let ones := Spec.fill (α := α) Numbers.one (.dim dim .scalar)
    { dim := dim, lo := ones, hi := ones }
  else
    { dim := dim
      lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
      hi := Spec.fill (α := α) Numbers.one (.dim dim .scalar) }

/-!
## Hard-masked softmax IBP (last axis)

Blocked coordinates have weight zero. An allowed coordinate lies in `[0,1]`, and it has weight one
when it is the only allowed coordinate in its row. These bounds do not evaluate `exp` or division,
so they remain valid for executable endpoint types whose `BoundOps` instance covers only directed
arithmetic. A tighter transfer rule requires separately certified directed bounds for
transcendental operations.
-/

/-- Conservative interval bounds for hard-masked softmax along the last tensor axis. -/
def ibpHardMaskedSoftmaxLastTensor : {s : Shape} →
    Tensor α s → Tensor α s → Tensor Bool s → (Tensor α s × Tensor α s)
  | .scalar, _lo, _hi, Tensor.scalar allowed =>
      let value := if allowed then Numbers.one else Numbers.zero
      (Tensor.scalar value, Tensor.scalar value)
  | .dim n .scalar, Tensor.dim _lo, Tensor.dim _hi, Tensor.dim allowed =>
      let lower := Tensor.dim fun i =>
        match allowed i with
        | Tensor.scalar false => Tensor.scalar Numbers.zero
        | Tensor.scalar true =>
            let hasOtherAllowed := (List.finRange n).any fun j =>
              i != j && match allowed j with
                | Tensor.scalar a => a
            Tensor.scalar (if hasOtherAllowed then Numbers.zero else Numbers.one)
      let upper := Tensor.dim fun i =>
        match allowed i with
        | Tensor.scalar true => Tensor.scalar Numbers.one
        | Tensor.scalar false => Tensor.scalar Numbers.zero
      (lower, upper)
  | .dim n inner, Tensor.dim lo, Tensor.dim hi, Tensor.dim allowed =>
      let lower := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmaxLastTensor (s := inner) (lo i) (hi i) (allowed i)).1
      let upper := Tensor.dim fun i : Fin n =>
        (ibpHardMaskedSoftmaxLastTensor (s := inner) (lo i) (hi i) (allowed i)).2
      (lower, upper)

/-!
## LayerNorm IBP (last axis)

Layer normalization (Ba et al.) computes, per vector, something like:

`y = (x - mean(x)) / sqrt(var(x) + eps)`.

We implement a conservative enclosure by:
1. Bounding mean using sums of endpoints.
2. Bounding variance using a max-deviation upper bound.
3. Bounding the per-component ratio by checking endpoint combinations against a positive denominator
   interval.

This is intended as a simple checker-side transfer rule. It is conservative and is not an
optimized relaxation.

References:
- Ba, Kiros, Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Bound propagation context: Xu et al., 2020 (auto_LiRPA): https://arxiv.org/abs/2002.12920
-/

/--
Ideal-arithmetic upper bound on the variance term used by analytic LayerNorm rules.

Given endpoint bounds for a vector and bounds on its mean, each coordinate is at most
`max |x_i - μ|` away from the bounded mean interval. Squaring and summing those coordinate radii
gives a conservative variance upper bound. The implementation uses ordinary scalar arithmetic and
is called only from exact-arithmetic branches; executable endpoint propagation uses
`ibpLayerNormRange?` instead.
-/
def idealLayerNormVarianceUpper {n : Nat}
    (lo hi : Tensor α [n]) (muLo muHi : α) : α :=
  if _h : n > 0 then
    match lo, hi with
    | .dim flo, .dim fhi =>
        let sumAbsSq : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let dl := MathFunctions.abs (l - muHi)
            let du := MathFunctions.abs (u - muLo)
            let a := if dl > du then dl else du
            acc + (a * a)) 0
        sumAbsSq / (n : Nat)
  else
    Numbers.zero

/-- Ideal-arithmetic mean bounds for a nonempty vector with bounded coordinates.

For `n = 0`, the mathematical mean is undefined; this total helper returns `(0,0)` so callers do
not accidentally divide by zero while they reject or totalize the empty case.
-/
def idealLayerNormMeanBounds {n : Nat}
    (lo hi : Tensor α [n]) : α × α :=
  if _h : n > 0 then
    let nA : α := (n : Nat)
    (Spec.Tensor.sumSpec lo / nA, Spec.Tensor.sumSpec hi / nA)
  else
    (Numbers.zero, Numbers.zero)

/--
Ideal-arithmetic bounds for `x - μ` when `x` and `μ` are bounded by intervals.

LayerNorm transfer rules repeatedly need this centered interval for the input, first derivative,
and second derivative streams. Keeping it here avoids duplicating the same endpoint arithmetic in
IBP and derivative propagation.
-/
def idealLayerNormCenteredBounds {n : Nat}
    (lo hi : Tensor α [n]) (muLo muHi : α) :
    Tensor α [n] × Tensor α [n] :=
  let flo := match lo with | .dim f => f
  let fhi := match hi with | .dim f => f
  let loOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl < du then dl else du))
  let hiOut :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let dl := l - muHi
        let du := u - muLo
        Tensor.scalar (if dl > du then dl else du))
  (loOut, hiOut)

/-- Ideal-arithmetic reciprocal-denominator bounds from an upper variance bound. -/
def idealLayerNormInvStdBounds (varHi : α) : α × α :=
  let sLo := MathFunctions.sqrt Numbers.epsilon
  let sHi := MathFunctions.sqrt (varHi + Numbers.epsilon)
  (Numbers.one / (if sHi > Numbers.epsilon then sHi else Numbers.epsilon),
   Numbers.one / (if sLo > Numbers.epsilon then sLo else Numbers.epsilon))

/-- Analytic real-arithmetic LayerNorm bounds on the last axis, lifted over leading dimensions. -/
def idealLayerNormLastTensor : {s : Shape} → Tensor α s → Tensor α s → (Tensor α s × Tensor α s)
  | .scalar, lo, hi => (lo, hi)
  | .dim n .scalar, lo, hi =>
      if n > 0 then
        let nA : α := (n : Nat)
        let sum_lo := Spec.Tensor.sumSpec lo
        let sum_hi := Spec.Tensor.sumSpec hi
        let mu_lo := sum_lo / nA
        let mu_hi := sum_hi / nA
        let flo := match lo with | .dim f => f
        let fhi := match hi with | .dim f => f
        let var_hi := idealLayerNormVarianceUpper (α := α) lo hi mu_lo mu_hi
        let den_lo := MathFunctions.sqrt Numbers.epsilon
        let den_hi := MathFunctions.sqrt (var_hi + Numbers.epsilon)
        let outLo :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              -- For positive denom interval [den_lo, den_hi], bound (x/denom) by checking all
              -- endpoint ratios.
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mn12 := if c1 < c2 then c1 else c2
              let mn34 := if c3 < c4 then c3 else c4
              let mn := if mn12 < mn34 then mn12 else mn34
              Tensor.scalar mn)
        let outHi :=
          Tensor.dim (fun i =>
            match flo i, fhi i with
            | .scalar l, .scalar u =>
              let dl := l - mu_hi
              let du := u - mu_lo
              let c1 := dl / den_lo
              let c2 := dl / den_hi
              let c3 := du / den_lo
              let c4 := du / den_hi
              let mx12 := if c1 > c2 then c1 else c2
              let mx34 := if c3 > c4 then c3 else c4
              let mx := if mx12 > mx34 then mx12 else mx34
              Tensor.scalar mx)
        (outLo, outHi)
      else
        -- Degenerate n=0: pass through
        (lo, hi)
  | .dim n inner, Tensor.dim loF, Tensor.dim hiF =>
      let outLo := Tensor.dim (fun i : Fin n => (idealLayerNormLastTensor (s := inner) (loF i) (hiF
        i)).1)
      let outHi := Tensor.dim (fun i : Fin n => (idealLayerNormLastTensor (s := inner) (loF i) (hiF
        i)).2)
      (outLo, outHi)

/--
Uniform finite enclosure for last-axis layer normalization.

For a row of length `n`, exact LayerNorm without affine scale or bias satisfies
`|yᵢ| ≤ sqrt n`; a singleton row is identically zero. Backends provide the outward-rounded bound
through `NonlinearBoundOps.layerNormAbsBound`. Returning `none` is preferable to evaluating the
normalization with unqualified host division and square root.
-/
def ibpLayerNormRange? [NonlinearBoundOps α]
    (s : Shape) (dim : Nat) : Option (FlatBox α) :=
  let rowLength := lastDimLen s
  if rowLength = 0 then
    some
      { dim := dim
        lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
        hi := Spec.fill (α := α) Numbers.zero (.dim dim .scalar) }
  else if rowLength = 1 then
    some
      { dim := dim
        lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
        hi := Spec.fill (α := α) Numbers.zero (.dim dim .scalar) }
  else do
    let radius ← NonlinearBoundOps.layerNormAbsBound (α := α) rowLength
    pure
      { dim := dim
        lo := Spec.fill (α := α) (-radius) (.dim dim .scalar)
        hi := Spec.fill (α := α) radius (.dim dim .scalar) }

/-- For tensors known to have shape `.dim n .scalar`, extract the underlying function. -/
@[expose] public def getDimScalarFn {n : Nat} (t : Tensor α [n]) : Fin n → Tensor α .scalar :=
  match t with
  | .dim f => f

-- Casting helpers for dependent shapes
/-- Cast a 1D `Box` along an equality of dimensions. -/
@[expose]
public def castBoxDim {n n' : Nat}
  (h : n = n')
  (B : Box α (.dim n .scalar)) : Box α (.dim n' .scalar) := by
  simpa [h] using B

/-- Cast a ReLU relaxation vector across a proven-equal hidden dimension. -/
def castRelax {n n' : Nat}
  (h : n = n')
  (r : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n]) :
  Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n'] := by
  simpa [h] using r

/-- Cast the input dimension of an affine map across a proven equality. -/
def castAffineIn {n n' m : Nat}
  (h : n = n') (a : AffineVec α n m) : AffineVec α n' m := by
  simpa [h] using a

/-- Cast the output dimension of an affine map across a proven equality. -/
@[expose]
def castAffineOut {n m m' : Nat}
  (h : m = m') (a : AffineVec α n m) : AffineVec α n m' := by
  simpa [h] using a

/--
Cast a dim-scalar tensor across an equality of dimensions.

We keep this as an `abbrev` so it unfolds aggressively in simp-based soundness proofs.
-/
abbrev castDimScalar {n n' : Nat}
    (h : n = n') (t : Tensor α [n]) : Tensor α [n'] :=
  Tensor.castShape t (congrArg (fun k => Shape.dim k Shape.scalar) h)

omit [Context α] [BoundOps α] in
/-- Casting a flat vector tensor along an equality from a dimension to itself changes no data. -/
@[simp] theorem castDimScalar_self {n : Nat}
    (h : n = n) (t : Tensor α [n]) :
    castDimScalar (α := α) h t = t := by
  exact Tensor.cast_shape_self t _

/-- IBP propagation through explicit linear parameters. -/
@[expose]
public def ibpLinearParams (p : LinParams α) (Xin : FlatBox α) : Option (FlatBox α) :=
  if h : Xin.dim = p.n then
    let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
    let bBox : Box α (.dim p.m .scalar) := Box.point (α:=α) p.b
    let yB   := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB bBox
    -- Materialize to avoid deep closure chains in multi-layer verifier runs.
    let yB' : Box α (.dim p.m .scalar) :=
      { lo := Tensor.materialize yB.lo
        hi := Tensor.materialize yB.hi }
    some (toFlatBox p.m yB')
  else none

/-- IBP propagation for a `.linear` node using `ParamStore.linearWB`. -/
@[expose]
public def ibpLinear (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.linearWB[id]? with
  | none => none
  | some p => ibpLinearParams (α := α) p Xin

/-- IBP propagation for a `.matmul` node (bias-free) using `ParamStore.matmulW`. -/
@[expose]
public def ibpMatmul (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.matmulW[id]? with
  | none => none
  | some p =>
    if h : Xin.dim = p.n then
      let xB   : Box α (.dim p.n .scalar) := castBoxDim (α:=α) h (ofFlatBox Xin)
      let zeroB : Box α (.dim p.m .scalar) :=
        let z := Spec.fill (α:=α) 0 (.dim p.m .scalar)
        Box.point (α:=α) z
      let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
      -- Materialize to avoid deep closure chains (runtime performance).
      let yB' : Box α (.dim p.m .scalar) :=
        { lo := Tensor.materialize yB.lo
          hi := Tensor.materialize yB.hi }
      some (toFlatBox p.m yB')
    else none

/--
IBP transfer for a supported convolution node whose parameters are stored in `ParamStore.convCfg`.
-/
def ibpConvNode (config : NN.IR.ConvConfig) (parentShape outShape : Shape)
    (id : Nat) (ps : ParamStore α) (Xin : FlatBox α) : Option (FlatBox α) :=
  match ps.convCfg[id]? with
  | none => none
  | some cfg =>
    if !convTransferSupported (α := α) config cfg parentShape outShape then
      none
    else
      let expected := cfg.inChannels * cfg.inputSpatial.toList.prod
      if hdim : Xin.dim = expected then
        let sFlat := Shape.dim Xin.dim Shape.scalar
        let sIn := Shape.ofList (cfg.inChannels :: cfg.inputSpatial.toList)
        have hsize : sFlat.size = sIn.size := by
          simp [Spec.Shape.size, sFlat, sIn, hdim, expected]
        let xLo := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.lo hsize
        let xHi := Tensor.reshapeSpec (α:=α) (s₁:=sFlat) (s₂:=sIn) Xin.hi hsize
        let xBox : Box α sIn := { lo := xLo, hi := xHi }
        let yBox := NN.MLTheory.CROWN.ibpConv (α:=α) (layer:=cfg.spec) (xB:=xBox)
        let outSpatial := Spec.convOutSpatial cfg.inputSpatial cfg.kernel cfg.stride cfg.padding
        let flatOutShape := Shape.ofList (cfg.outChannels :: outSpatial.toList)
        let flatLo := Tensor.flattenSpec (α:=α) yBox.lo
        let flatHi := Tensor.flattenSpec (α:=α) yBox.hi
        some { dim := flatOutShape.size, lo := flatLo, hi := flatHi }
      else none

/-- Apply a monotone `SomeTensor` operation independently to both interval endpoints. -/
def ibpMonotoneSomeTensor?
    (parentShape outShape : Shape)
    (op : SomeTensor α → Except String (SomeTensor α))
    (input : FlatBox α) : Option (FlatBox α) :=
  if hdim : input.dim = parentShape.size then
    let flatShape := Shape.dim input.dim Shape.scalar
    have hsize : flatShape.size = parentShape.size := by
      simp [flatShape, Shape.size, hdim]
    let loInput := Tensor.reshapeSpec (α := α) (s₁ := flatShape) (s₂ := parentShape) input.lo hsize
    let hiInput := Tensor.reshapeSpec (α := α) (s₁ := flatShape) (s₂ := parentShape) input.hi hsize
    match op (SomeTensor.mk (α := α) parentShape loInput),
        op (SomeTensor.mk (α := α) parentShape hiInput) with
    | .ok lo, .ok hi =>
        if hlo : lo.shape = outShape then
          if hhi : hi.shape = outShape then
            let loTensor : Tensor α outShape := hlo ▸ lo.tensor
            let hiTensor : Tensor α outShape := hhi ▸ hi.tensor
            some
              { dim := outShape.size
                lo := Tensor.flattenSpec (α := α) loTensor
                hi := Tensor.flattenSpec (α := α) hiTensor }
          else
            none
        else
          none
    | _, _ => none
  else
    none


end NN.MLTheory.CROWN.Graph
