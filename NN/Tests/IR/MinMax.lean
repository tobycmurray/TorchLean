/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Infer
public import NN.Spec.Layers.Activation
public meta import NN.IR.Infer
public meta import NN.Spec.Layers.Activation
public meta import NN.Spec.Core.Shape
public import NN.MLTheory.CROWN.Graph.Engine.Base
public meta import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# MinMax tests

`OpKind.minMax` pairs adjacent entries along an axis and replaces each pair by
its minimum and its maximum, smaller first.  These checks pin the pairing
convention, the behaviour on each axis of a rank-2 tensor, the totality of the
spec at an odd extent, and the shape contract that rejects that odd extent at
the graph level.
-/

@[expose] public section

namespace NN.Tests.IR.MinMax

open NN
open NN.IR
open _root_.Spec
open scoped NN.Spec.RationalAlgebraic

private def vec {n : Nat} (xs : Array Int) : Tensor Int [n] :=
  Tensor.dim fun i => Tensor.scalar (xs.getD i.val 0)

private def readVec {n : Nat} (t : Tensor Int [n]) : List Int :=
  (List.finRange n).map fun i => ((Tensor.dimEquiv n .scalar t) i).item

private def mat (xs : Array (Array Int)) : Tensor Int [2, 4] :=
  Tensor.dim fun r => Tensor.dim fun c => Tensor.scalar ((xs.getD r.val #[]).getD c.val 0)

private def readMat (t : Tensor Int [2, 4]) : List (List Int) :=
  (List.finRange 2).map fun r =>
    readVec ((Tensor.dimEquiv 2 (.dim 4 .scalar) t) r)

-- Pairs are adjacent and non-overlapping: `(3,1)` and `(2,5)`.
/-- info: [1, 3, 2, 5] -/
#guard_msgs in
#eval readVec (Activation.minMaxOuterSpec (α := Int) (vec (n := 4) #[3, 1, 2, 5]))

-- On the innermost axis of a `[2,4]`, each row is paired independently.
/-- info: [[2, 9, 7, 7], [-1, 0, 3, 4]] -/
#guard_msgs in
#eval readMat (Activation.minMaxAxisSpec (α := Int) 1 (mat #[#[9, 2, 7, 7], #[0, -1, 4, 3]]))

-- On the outermost axis the two rows are paired elementwise.
/-- info: [[0, -1, 4, 3], [9, 2, 7, 7]] -/
#guard_msgs in
#eval readMat (Activation.minMaxAxisSpec (α := Int) 0 (mat #[#[9, 2, 7, 7], #[0, -1, 4, 3]]))

-- An odd extent leaves the trailing entry alone, so the spec is total. A graph with an odd axis is rejected before this can be reached.
/-- info: [1, 8, 6] -/
#guard_msgs in
#eval readVec (Activation.minMaxOuterSpec (α := Int) (vec (n := 3) #[8, 1, 6]))

/-- The pairing is an involution on indices. -/
example : ∀ i < 4, Activation.minMaxPartner 4 (Activation.minMaxPartner 4 i) = i := by decide

-- The shape contract accepts an even axis.
/-- info: Except.ok () -/
#guard_msgs in
#eval OpContracts.checkMinMaxAxis 0 (Shape.dim 4 (.dim 3 .scalar))

-- …and rejects an odd one, rather than silently leaving an entry unpaired.
/-- info: Except.error "min_max: axis 0 has odd extent 5; MinMax pairs entries up" -/
#guard_msgs in
#eval OpContracts.checkMinMaxAxis 0 (Shape.dim 5 (.dim 3 .scalar))

-- An out-of-range axis is rejected too.
/-- info: Except.error "invalid axis 3 for rank 2" -/
#guard_msgs in
#eval OpContracts.checkMinMaxAxis 3 (Shape.dim 4 (.dim 3 .scalar))

/-! ### The CROWN interval bounds

The engines flatten, but the node keeps its shape and channel axis, so the
pairing survives as a stride and an extent.  `boxMinMax` is exact: every corner
of the output box is attained, because `min` and `max` are monotone in both
arguments.
-/

open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph in
private def box : FlatBox ℚ :=
  { dim := 4
    lo := Tensor.dim fun i => Tensor.scalar (#[(-2 : ℚ), 1, 0, 5].getD i.val 0)
    hi := Tensor.dim fun i => Tensor.scalar (#[(3 : ℚ), 4, 7, 6].getD i.val 0) }

private def readQ {n : Nat} (t : Tensor ℚ [n]) : List ℚ :=
  (List.finRange n).map fun i => ((Tensor.dimEquiv n .scalar t) i).item

-- Pairs (0,1) and (2,3): the min entry takes both componentwise minima.
/-- info: [-2, 1, 0, 5] -/
#guard_msgs in
#eval readQ (NN.MLTheory.CROWN.Graph.boxMinMax (α := ℚ) 1 4 box).lo

/-- info: [3, 4, 6, 7] -/
#guard_msgs in
#eval readQ (NN.MLTheory.CROWN.Graph.boxMinMax (α := ℚ) 1 4 box).hi

-- Axis 0 of a `[2,3]` has extent 2 and stride 3; axis 1 has extent 3, stride 1.
/-- info: (2, 3) -/
#guard_msgs in
#eval (NN.MLTheory.CROWN.Graph.minMaxFlatExtent 0 (Shape.dim 2 (.dim 3 .scalar)),
       NN.MLTheory.CROWN.Graph.minMaxFlatStride 0 (Shape.dim 2 (.dim 3 .scalar)))

/-- info: (3, 1) -/
#guard_msgs in
#eval (NN.MLTheory.CROWN.Graph.minMaxFlatExtent 1 (Shape.dim 2 (.dim 3 .scalar)),
       NN.MLTheory.CROWN.Graph.minMaxFlatStride 1 (Shape.dim 2 (.dim 3 .scalar)))

-- Pairing a `[2,3]` along axis 0 swaps the two rows of three.
/-- info: [3, 4, 5, 0, 1, 2] -/
#guard_msgs in
#eval (List.range 6).map (NN.MLTheory.CROWN.Graph.minMaxFlatPartner 3 2)

end NN.Tests.IR.MinMax
