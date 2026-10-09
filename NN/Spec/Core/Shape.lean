/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Basic

import Init.Grind
import Mathlib.Data.Nat.Init

/-!
# Shapes (`Spec.Shape`)

`Shape` is the type-level “shape descriptor” for tensors in the spec layer.

TorchLean uses *shape-indexed tensors*:

`Tensor α s`

so `Shape` is how we encode the structure of `s` in a way Lean can use for both computation and
proofs.

## Representation

`Shape` is an inductive tree:

- `.scalar`
- `.dim n s`  (a length-`n` dimension whose entries have shape `s`)

This matches the tensor definition in `NN/Spec/Core/Tensor/Core.lean`.

## Common utilities

- `Spec.Shape.size : Shape → Nat` is the total number of scalar elements (“numel”).
- `Shape.toList : Shape → List Nat` is a convenient runtime view used by front-ends and bridges.

PyTorch analogy:

- `Shape.toList s` corresponds to `tensor.shape` (a tuple of dimensions).
- `Spec.Shape.rank s` corresponds to `tensor.ndim`.
- `Spec.Shape.size s` corresponds to `tensor.numel()`.

## Broadcasting and axes

Broadcasting is encoded via `CanBroadcastTo` / `BroadcastTo`.

This is an intentionally *asymmetric* relation ("broadcast `s1` to `s2`"), because most tensor code
is naturally written by choosing the output shape and requiring each input to broadcast to it.

The typeclass wrapper `BroadcastTo` keeps higher-level specs readable: in many cases Lean can infer
the broadcast evidence automatically, so call sites don’t have to manually thread proofs around.

It also defines axis-validity helpers (`NonemptyAxis`) and a `well_formed` predicate for “all
dimensions are positive”, which is useful when you want to rule out degenerate cases in proofs.
-/

@[expose] public section


namespace Spec

/-!
We represent shapes as an inductive tree instead of a bare `List Nat` because:

- it matches the tensor representation (`Tensor α s`) structurally, so many definitions are simple
  structural recursion,
- it keeps "scalar vs dim" cases explicit (important for proofs),
- it gives definitional equalities that are friendlier than lists in many places.
-/
/--
Tensor shape descriptor used to index spec-level tensors (`Spec.Tensor α s`).

`Shape` is an outermost-first tree:
- `.scalar` for a scalar,
- `.dim n s` for a length-`n` dimension whose entries have shape `s`.
-/
inductive Shape where
  | scalar : Shape
  | dim : Nat → Shape → Shape
deriving DecidableEq, Repr

/-!
Model code writes shapes as dimension lists. For example, `Tensor Float [4, 2]` is a four-by-two
tensor of Lean `Float` values, while `Trainer.Dataset [2] [1]` describes supervised samples with
two input values and one target value. The expected `Shape` type directs Lean to elaborate the list
through `Shape.ofList`.

The brackets are shape notation, not tensor storage: a value of type `Tensor Float [4, 2]` is still
a `Tensor`, never a `List`. During elaboration the notation reduces to the recursive shape below,
preserving the definitional equalities used by tensor programs and proofs. Ordinary model code
should not write the expanded `.dim` form.
-/
namespace Shape

/--
Output length of a floor-mode sliding window with symmetric padding.

For positive `kernel` and `stride` with
$\mathtt{kernel}\le\mathtt{input}+2\mathtt{padding}$, this is
$(\mathtt{input}+2\mathtt{padding}-\mathtt{kernel})/\mathtt{stride}+1$. Invalid geometry has length zero, so saturated
natural-number subtraction and division by zero cannot create a phantom output element.
-/
def slidingWindowOutDim (input kernel stride padding : Nat) : Nat :=
  let padded := input + 2 * padding
  if kernel = 0 || stride = 0 || padded < kernel then
    0
  else
    (padded - kernel) / stride + 1

/-- Build a shape from a list of dimensions (outermost first). -/
abbrev ofList : List Nat → Shape
  | [] => .scalar
  | n :: ns => .dim n (ofList ns)

/-- Build a shape from runtime dimensions stored outermost first. -/
def ofArray (dims : Array Nat) : Shape :=
  dims.foldr (fun extent rest => .dim extent rest) .scalar

/-- Display list-shaped dimensions with the same notation accepted by model code. -/
@[app_unexpander Spec.Shape.ofList]
meta def ofListUnexpander : Lean.PrettyPrinter.Unexpander
  | `($_ $dims) => pure dims
  | _ => throw ()

/-- Interpret a dimension list as a tensor shape. -/
instance : Coe (List Nat) Shape where
  coe := ofList

/-- Convert to a list of dimensions, outermost first. -/
def toList : Shape → List Nat
  | .scalar => []
  | .dim n rest => n :: toList rest

/-- Print a shape using the same dimension-list convention as model code. -/
def pretty (s : Shape) : String :=
  "[" ++ String.intercalate ", " (s.toList.map toString) ++ "]"

/-- Swap two adjacent dimensions at a given depth (0‑based from the outermost). -/
@[reducible] def swapAdjacentAtDepth (s : Shape) (depth : Nat) : Shape :=
  match depth, s with
  | 0, .dim m (.dim n rest) => .dim n (.dim m rest)
  | d+1, .dim m rest => .dim m (swapAdjacentAtDepth rest d)
  | _, _ => s  -- invalid depth, return unchanged

@[simp] theorem swapAdjacentAtDepth_zero_rank_two (m n : Nat) :
    ([m, n] : Shape).swapAdjacentAtDepth 0 = [n, m] := rfl

/-- Swapping adjacent dims at depth `depth` twice returns the original shape. -/
@[simp] theorem swapAdjacentAtDepth_involutive (s : Shape) (depth : Nat) :
    (s.swapAdjacentAtDepth depth).swapAdjacentAtDepth depth = s := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => simp [swapAdjacentAtDepth]
      | dim m rest =>
          cases rest <;> simp [swapAdjacentAtDepth]
  | succ d ih =>
      cases s <;> simp [swapAdjacentAtDepth, ih]

/-- Shape obtained by applying adjacent-axis swaps from left to right. -/
def applyAdjacentSwaps : Shape → List Nat → Shape
  | s, [] => s
  | s, depth :: depths => applyAdjacentSwaps (s.swapAdjacentAtDepth depth) depths

/-- Adjacent swaps that move `axis` to the innermost position of a rank-`rank` shape. -/
def moveAxisToInnermostSwaps (rank axis : Nat) : List Nat :=
  (List.range (rank - (axis + 1))).map (axis + ·)

@[simp]
theorem applyAdjacentSwaps_append (s : Shape) (xs ys : List Nat) :
    applyAdjacentSwaps s (xs ++ ys) = applyAdjacentSwaps (applyAdjacentSwaps s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.cons_append, applyAdjacentSwaps]
      exact ih (s.swapAdjacentAtDepth depth)

/-- Replaying adjacent-axis swaps in reverse order restores the original shape. -/
@[simp]
theorem applyAdjacentSwaps_reverse (s : Shape) (depths : List Nat) :
    applyAdjacentSwaps (applyAdjacentSwaps s depths) depths.reverse = s := by
  induction depths generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [applyAdjacentSwaps, List.reverse_cons, applyAdjacentSwaps_append]
      rw [ih]
      exact swapAdjacentAtDepth_involutive s depth

/-- Append a new innermost dimension. -/
@[reducible]
def appendDim (s : Shape) (n : Nat) : Shape :=
  match s with
  | .scalar => .dim n .scalar
  | .dim m rest => .dim m (appendDim rest n)

/-- Add a new outermost dimension. -/
@[reducible]
def prependDim (s : Shape) (n : Nat) : Shape :=
  .dim n s

/-- Concatenate two shapes, preserving the dimensions of the first shape as leading axes. -/
@[reducible]
def concat : Shape → Shape → Shape
  | .scalar, suffix => suffix
  | .dim n rest, suffix => .dim n (concat rest suffix)

/-- View a shape as `leading.concat (.dim extent suffix)`, where `leading` holds
exactly the `axis` dimensions in front of the named axis.

This is what lets an operation on an arbitrary axis be written with the
structural primitive for the *leading* axis, rather than by permuting the axis
to the front and back again. -/
def splitAtAxis : Nat → Shape → Option (Shape × Nat × Shape)
  | _, .scalar => none
  | 0, .dim n suffix => some (.scalar, n, suffix)
  | k + 1, .dim m rest =>
      match splitAtAxis k rest with
      | some (leading, n, suffix) => some (.dim m leading, n, suffix)
      | none => none

/-- The split is a decomposition: putting the pieces back gives the shape. -/
theorem splitAtAxis_concat :
    ∀ (axis : Nat) (s leading suffix : Shape) (n : Nat),
      splitAtAxis axis s = some (leading, n, suffix) →
      leading.concat (.dim n suffix) = s
  | _, .scalar, _, _, _, h => by simp [splitAtAxis] at h
  | 0, .dim m rest, leading, suffix, n, h => by
      simp only [splitAtAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      rfl
  | k + 1, .dim m rest, leading, suffix, n, h => by
      simp only [splitAtAxis] at h
      match hrec : splitAtAxis k rest with
      | none => rw [hrec] at h; exact absurd h (by simp)
      | some (l, n', s') =>
          rw [hrec] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl, rfl⟩ := h
          exact congrArg (Shape.dim m) (splitAtAxis_concat k rest l s' n' hrec)

/-- A split names the extent the axis actually has. -/
theorem splitAtAxis_extent :
    ∀ (axis : Nat) (s leading suffix : Shape) (n : Nat),
      splitAtAxis axis s = some (leading, n, suffix) → s.toList[axis]? = some n
  | _, .scalar, _, _, _, h => by simp [splitAtAxis] at h
  | 0, .dim m rest, leading, suffix, n, h => by
      simp only [splitAtAxis, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      rfl
  | k + 1, .dim m rest, leading, suffix, n, h => by
      simp only [splitAtAxis] at h
      match hrec : splitAtAxis k rest with
      | none => rw [hrec] at h; exact absurd h (by simp)
      | some (l, n', s') =>
          rw [hrec] at h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl, rfl⟩ := h
          simpa [toList] using splitAtAxis_extent k rest l s' n' hrec

/-- Shape concatenation is associative. -/
@[simp] theorem concat_assoc (left middle right : Shape) :
    (left.concat middle).concat right = left.concat (middle.concat right) := by
  induction left with
  | scalar => rfl
  | dim n rest ih => simp only [concat, ih]

/-- Converting an appended dimension list is shape concatenation. -/
@[simp] theorem ofList_append (left right : List Nat) :
    ofList (left ++ right) = (ofList left).concat (ofList right) := by
  induction left with
  | nil => rfl
  | cons n left ih => simp [ofList, concat, ih]

/-- The list view of concatenated shapes is the concatenation of their list views. -/
@[simp] theorem toList_concat (left right : Shape) :
    (left.concat right).toList = left.toList ++ right.toList := by
  induction left with
  | scalar => rfl
      | dim n rest ih => simp [toList, ih]

/-- Appending one dimension is concatenation with a one-axis suffix. -/
theorem appendDim_eq_concat (s : Shape) (n : Nat) :
    s.appendDim n = s.concat (.dim n .scalar) := by
  induction s with
  | scalar => rfl
  | dim m rest ih => simp only [appendDim, concat, ih]

/-- Appending two dimensions is concatenation with a two-axis suffix. -/
theorem appendDim_appendDim_eq_concat (s : Shape) (m n : Nat) :
    (s.appendDim m).appendDim n = s.concat (.dim m (.dim n .scalar)) := by
  induction s with
  | scalar => rfl
  | dim k rest ih => simp only [appendDim, concat, ih]

/-- Appending a final dimension commutes with adding a fixed leading shape. -/
@[simp]
theorem concat_appendDim (leading suffix : Shape) (n : Nat) :
    (leading.concat suffix).appendDim n = leading.concat (suffix.appendDim n) := by
  induction leading with
  | scalar => rfl
  | dim m rest ih =>
      simp only [concat, appendDim, ih]

/-- Total number of scalar elements (a.k.a. “numel”). -/
def size : Shape → Nat
  | .scalar => 1
  | .dim n rest => n * size rest

/-- The number of entries in a list-shaped tensor is the product of its dimensions. -/
@[simp] theorem size_ofList (dims : List Nat) :
    size (ofList dims) = dims.prod := by
  induction dims with
  | nil => rfl
  | cons dim dims ih => simp [size, ih]

/-- A shape consisting only of singleton axes contains one scalar. -/
@[simp] theorem size_ofList_replicate_one (n : Nat) :
    size (ofList (List.replicate n 1)) = 1 := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [List.replicate_succ]
      simp [size, ih]

/--
`appendDim` multiplies the number of scalar elements by the appended dimension.

This lemma is the standard justification for reshape tricks where we:
- treat a tensor of shape `s.appendDim n` as a matrix of shape `(size s) × n`, or
- append an extra singleton dimension (`n = 1`) without changing `size`.
-/
theorem size_appendDim (s : Shape) (n : Nat) : size (appendDim s n) = size s * n := by
  induction s with
  | scalar =>
      simp [size]
  | dim m rest ih =>
      -- `appendDim` recurses to the innermost dimension; `size` is multiplicative.
      simp [size, ih, Nat.mul_assoc]

/-- The number of elements in a concatenated shape is the product of the two shape sizes. -/
theorem size_concat (leading suffix : Shape) :
    size (concat leading suffix) = size leading * size suffix := by
  induction leading with
  | scalar => simp [size]
  | dim n rest ih => simp [size, ih, Nat.mul_assoc]

/-- `ofList` is a left inverse of `toList`. -/
@[simp] theorem ofList_toList (s : Shape) : ofList (toList s) = s := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
    simp [toList, ih]

/-- `toList` is a right inverse of `ofList`. -/
@[simp] theorem toList_ofList (xs : List Nat) : toList (ofList xs) = xs := by
  induction xs with
  | nil => rfl
  | cons n ns ih =>
    simp [toList, ih]

-- Tell `grind` about the standard shape normalization lemmas.
attribute [grind =] size_appendDim size_ofList ofList_toList toList_ofList

/-- Convert to an array of dimensions (outermost first). -/
def toArray (s : Shape) : Array Nat :=
  toList s |>.toArray

/-- Boolean equality test for shapes (structural). -/
def areEqual : Shape → Shape → Bool
  | .scalar, .scalar => true
  | .dim n1 s1, .dim n2 s2 => n1 == n2 && areEqual s1 s2
  | _, _ => false

-- We keep `BEq` as an explicit structural test because it shows up in runtime checks and logs.
/-- `BEq Shape` uses the explicit structural boolean test `Shape.areEqual`. -/
instance : BEq Shape where
  beq := areEqual

/-- Default inhabitant for `Shape`, used only when Lean needs a canonical fallback value. -/
instance : Inhabited Shape where
  default := .scalar

/-- Get dimension at index `i` (0‑based), or `none` if out of bounds. -/
def getDim : Shape → Nat → Option Nat
  | .scalar, _ => none
  | .dim n _, 0 => some n
  | .dim _ rest, i+1 => getDim rest i

/- Broadcasting support -/
/-!
### Typeclass-friendly broadcasting (`BroadcastTo`)

The `CanBroadcastTo` relation is asymmetric (“broadcast `s₁` *to* `s₂`”), matching how most
operations are written: we pick a target shape and require each operand to broadcast to it.

The `BroadcastTo` wrapper lets Lean search for a broadcast proof automatically, which is convenient
for higher-level specs (layers/models) where the broadcasting details are not the point.

PyTorch analogy:

- PyTorch broadcasting aligns shapes from the *trailing* dimensions by implicitly prepending `1`s
  to the shorter shape.
- Our `Shape` is an outermost-first tree, so the corresponding operation is `expand_dims`:
  it inserts leading/outer dimensions to reach the target rank (this is the "prepend `1`s" step).
- `dim_1_to_n` corresponds to PyTorch's "dimension 1 can expand to n" rule.
-/

/-- Rank = number of dimensions (scalar has rank 0). -/
def rank : Shape → Nat
  | Shape.scalar => 0
  | Shape.dim _ rest => 1 + rank rest

/-- Insert a dimension at an axis, where axis `0` is outermost.

Callers that construct a tensor of this shape also carry a proof that the axis does not exceed the
input rank. The out-of-bounds scalar case is therefore unreachable in typed tensor operations.
-/
@[reducible] def insertAxis : Shape → Nat → Nat → Shape
  | shape, axis, extent =>
      match axis with
      | 0 => .dim extent shape
      | axis + 1 =>
          match shape with
          | .dim n rest => .dim n (insertAxis rest axis extent)
          | .scalar => .scalar

/-- Inserting at axis zero adds a new outermost dimension. -/
@[simp] theorem insertAxis_zero (shape : Shape) (extent : Nat) :
    insertAxis shape 0 extent = .dim extent shape := by
  cases shape <;> rfl

/-- Appending one dimension increases the rank by one. -/
@[simp] theorem rank_appendDim (s : Shape) (n : Nat) :
    rank (s.appendDim n) = rank s + 1 := by
  induction s with
  | scalar => rfl
  | dim _ rest ih => simp only [rank, ih, Nat.add_assoc]

/-- The list view contains one entry for each tensor axis. -/
@[simp] theorem length_toList (s : Shape) : s.toList.length = s.rank := by
  induction s with
  | scalar => rfl
  | dim n rest ih => simp [toList, rank, ih, Nat.add_comm]

/-- A shape decomposed into a leading prefix and a suffix of a prescribed rank. -/
structure SuffixSplit (shape : Shape) (suffixRank : Nat) where
  /-- Axes preceding the suffix. -/
  leading : Shape
  /-- Extents of the suffix axes. -/
  suffix : List Nat
  /-- The suffix has the requested number of axes. -/
  suffix_length : suffix.length = suffixRank
  /-- The decomposition reconstructs the original shape. -/
  concat_eq : leading.concat (ofList suffix) = shape

/-- Split the final `suffixRank` axes from a shape. -/
def splitSuffix (shape : Shape) (suffixRank : Nat) (h : suffixRank ≤ shape.rank) :
    SuffixSplit shape suffixRank := by
  let dims := shape.toList
  have hdims : suffixRank ≤ dims.length := by simpa [dims] using h
  let split := dims.length - suffixRank
  let suffixDims := dims.drop split
  have hsuffix : suffixDims.length = suffixRank := by
    simp [suffixDims, split, List.length_drop, Nat.sub_sub_self hdims]
  refine
    { leading := ofList (dims.take split)
      suffix := suffixDims
      suffix_length := hsuffix
      concat_eq := ?_ }
  rw [← ofList_append]
  simp [split, suffixDims, dims]

/-- Swap the first two axes after an arbitrary fixed leading shape. -/
@[simp]
theorem swapAdjacentAtDepth_concat_rank (leading suffix : Shape) (m n : Nat) :
    swapAdjacentAtDepth (leading.concat (.dim m (.dim n suffix))) leading.rank =
      leading.concat (.dim n (.dim m suffix)) := by
  induction leading with
  | scalar => rfl
  | dim _ tail ih =>
      simp only [concat, rank, Nat.one_add, swapAdjacentAtDepth, ih]

/-- Replace every dimension by one while preserving the rank of a shape. -/
def singletonAxes : Shape → Shape
  | .scalar => .scalar
  | .dim _ rest => .dim 1 (singletonAxes rest)

@[simp] theorem rank_singletonAxes (s : Shape) : rank (singletonAxes s) = rank s := by
  induction s <;> simp [singletonAxes, rank, *]

@[simp] theorem size_singletonAxes (s : Shape) : size (singletonAxes s) = 1 := by
  induction s <;> simp [singletonAxes, size, *]

/-- Proposition used by broadcast constructors that align two existing dimensions. -/
class SameRank (s₁ s₂ : Shape) : Prop where
  /-- The two shapes have the same number of dimensions. -/
  rank_eq : rank s₁ = rank s₂

instance : SameRank .scalar .scalar := ⟨rfl⟩

instance sameRankRefl (s : Shape) : SameRank s s := ⟨rfl⟩

instance {s₁ s₂ : Shape} {n₁ n₂ : Nat} [tail : SameRank s₁ s₂] :
    SameRank (.dim n₁ s₁) (.dim n₂ s₂) :=
  ⟨by simpa [rank] using congrArg Nat.succ tail.rank_eq⟩

/-- Evidence that shape `s₁` can be broadcast to shape `s₂` using right-aligned dimensions.

The equal-dimension constructors require equal-rank tails. Consequently, every rank difference is
resolved by `expand_dims` before dimensions are compared, matching NumPy and PyTorch rather than
allowing a proof term to choose between left- and right-aligned interpretations. -/
inductive CanBroadcastTo : Shape → Shape → Type where
  /-- Scalar shapes agree. Higher-rank scalar broadcasts are built with `expand_dims`. -/
  | scalar : CanBroadcastTo .scalar .scalar
  /-- Matching outer dimensions preserve broadcasting of their tails. -/
  | dim_eq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
      (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim n s₁) (.dim n s₂)
  /-- An outer dimension of length one can expand to any target length. -/
  | dim_1_to_n {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
      (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim 1 s₁) (.dim n s₂)
  /-- A new outer target dimension aligns a source of lower rank. -/
  | expand_dims {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo s₁ (Shape.dim n s₂)
deriving Repr

/-- Every shape broadcasts to itself without expanding an axis. -/
def CanBroadcastTo.refl : (s : Shape) → CanBroadcastTo s s
  | .scalar => .scalar
  | .dim _ tail => .dim_eq (refl tail)

/-- The canonical witness that broadcasts a scalar by inserting every target dimension. -/
def CanBroadcastTo.scalarTo : (s : Shape) → CanBroadcastTo .scalar s
  | .scalar => .scalar
  | .dim _ tail => .expand_dims (scalarTo tail)

/-- Broadcast a shape of singleton axes to any shape of the same rank. -/
def CanBroadcastTo.singletonAxes : (s : Shape) → CanBroadcastTo (Shape.singletonAxes s) s
  | .scalar => .scalar
  | .dim _ tail =>
      letI : SameRank (Shape.singletonAxes tail) tail := ⟨rank_singletonAxes tail⟩
      .dim_1_to_n (singletonAxes tail)

/-- Broadcast a suffix across an arbitrary collection of newly prepended target dimensions. -/
def CanBroadcastTo.prependTarget : (leading suffix : Shape) →
    CanBroadcastTo suffix (Shape.concat leading suffix)
  | .scalar, suffix => refl suffix
  | .dim _ tail, suffix => .expand_dims (prependTarget tail suffix)

/-- Typeclass wrapper for `CanBroadcastTo` so broadcast proofs can be inferred. -/
class BroadcastTo (s₁ s₂ : Shape) where
  proof : CanBroadcastTo s₁ s₂

/-- Scalar shapes broadcast directly. Leading target dimensions are inferred by `expand_dims`. -/
instance broadcastToScalar : BroadcastTo Shape.scalar Shape.scalar where
  proof := CanBroadcastTo.scalar

/-- Broadcasting preserves equal leading dimensions when the tails broadcast. -/
instance broadcastToDimEq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim n s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_eq bc.proof

/-- Dimension `1` can broadcast to any `n` (PyTorch's main broadcast rule). -/
instance broadcastToDim1ToN {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim 1 s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_1_to_n bc.proof

/-- Prepend an outer dimension (the "expand_dims" step used to align ranks). -/
instance broadcastToExpandDims {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] :
    BroadcastTo s₁ (Shape.dim n s₂) where
  proof := CanBroadcastTo.expand_dims bc.proof

/-- Swap adjacent entries in an axis-ordering list, leaving invalid positions unchanged. -/
def swapAdjacentAxes (axes : List Nat) (depth : Nat) : List Nat :=
  match axes, depth with
  | [], _ => []
  | [axis], _ => [axis]
  | first :: second :: rest, 0 => second :: first :: rest
  | first :: rest, depth + 1 => first :: swapAdjacentAxes rest depth

/-- Permute axes of a shape using a zero-based structural axis ordering. Returns `none` if invalid. -/
def permute? (s : Shape) (perm : List Nat) : Option Shape :=
  let r := rank s
  if perm.length != r then
    none
  else if !(decide perm.Nodup) then
    none
  else
    let dims := toList s
    (perm.mapM fun i => dims[i]?).map ofList

/-- Axis permutation that exchanges `axis₁` and `axis₂` and fixes every other axis. -/
def transposePermutation (rank axis₁ axis₂ : Nat) : List Nat :=
  (List.range rank).map fun axis =>
    if axis = axis₁ then axis₂ else if axis = axis₂ then axis₁ else axis

/-- Remove one axis from a shape. Invalid axes leave the shape unchanged. -/
def eraseAxis : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ rest, 0 => rest
  | .dim n rest, axis + 1 => .dim n (eraseAxis rest axis)

/-- Replace the extent of one axis. Invalid axes leave the shape unchanged. -/
def replaceAxis : Shape → Nat → Nat → Shape
  | .scalar, _, _ => .scalar
  | .dim _ rest, 0, extent => .dim extent rest
  | .dim n rest, axis + 1, extent => .dim n (replaceAxis rest axis extent)

/-- Erasing an in-bounds axis decreases the rank by one. -/
theorem rank_eraseAxis {s : Shape} {axis : Nat} (h : axis < s.rank) :
    (s.eraseAxis axis).rank = s.rank - 1 := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => simp [eraseAxis, rank]
      | succ axis =>
          have hAxis : axis < rest.rank := by
            simp only [rank] at h
            grind
          simp only [eraseAxis, rank, ih hAxis]
          grind

/-- Replacing an in-bounds axis preserves rank. -/
theorem rank_replaceAxis {s : Shape} {axis extent : Nat} (h : axis < s.rank) :
    (s.replaceAxis axis extent).rank = s.rank := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => simp [replaceAxis, rank]
      | succ axis =>
          have hAxis : axis < rest.rank := by
            simp only [rank] at h
            grind
          simp [replaceAxis, rank, ih hAxis]

/-!
## Axis evidence

Axes are zero-based natural numbers. `AxisInBounds axis s` says only that the axis exists;
`HasNonemptyAxis axis s` additionally says that its extent is positive. Shape-preserving operations
such as softmax need the first condition. Reductions whose definition selects an element, such as
maximum and minimum, use the second.

Negative axes are normalized by frontends before reaching this layer. The innermost axis of a
positive-rank shape is `s.rank - 1`.
-/

/-- The zero-based `axis` names a dimension of `s`. Its extent may be zero. -/
class AxisInBounds (axis : Nat) (s : Shape) : Prop where
  /-- The axis is strictly smaller than the shape rank. -/
  proof : axis < rank s

/-- Looking up an axis below the rank of a shape succeeds. -/
theorem getDim_isSome_of_lt {s : Shape} {axis : Nat} (h : axis < rank s) :
    (getDim s axis).isSome = true := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          simp only [getDim]
          apply ih
          grind [rank]

/-- Extent of a statically valid axis. -/
def axisSize (s : Shape) (axis : Nat) [h : AxisInBounds axis s] : Nat :=
  (getDim s axis).get (getDim_isSome_of_lt h.proof)

/-- The outermost dimension is axis zero, regardless of its extent. -/
instance axisInBoundsZero {n s} : AxisInBounds 0 (.dim n s) :=
  ⟨by simpa [rank, Nat.add_comm] using Nat.zero_lt_succ s.rank⟩

/-- An inner axis remains in bounds under an additional outer dimension. -/
instance axisInBoundsSucc {n s axis} [h : AxisInBounds axis s] :
    AxisInBounds (axis + 1) (.dim n s) :=
  ⟨by simpa [rank, Nat.add_comm] using Nat.add_lt_add_right h.proof 1⟩

/-- The extent of the leading axis is its outer dimension. -/
@[simp] theorem axisSize_zero (n : Nat) (s : Shape) : axisSize (.dim n s) 0 = n := by
  rfl

/-- Looking through an outer dimension preserves the extent of an inner axis. -/
@[simp] theorem axisSize_succ (n : Nat) (s : Shape) (axis : Nat)
    [AxisInBounds axis s] [AxisInBounds (axis + 1) (.dim n s)] :
    axisSize (.dim n s) (axis + 1) = axisSize s axis := by
  simp only [axisSize, getDim]

/-- Decide whether a natural number names a dimension of `s`, returning typed evidence. -/
def axisInBounds? (axis : Nat) (s : Shape) : Option (PLift (AxisInBounds axis s)) :=
  if h : axis < rank s then some ⟨⟨h⟩⟩ else none

/-- The decision procedure succeeds whenever static in-bounds evidence is available. -/
theorem axisInBounds?_isSome {axis : Nat} {s : Shape} [h : AxisInBounds axis s] :
    (axisInBounds? axis s).isSome := by
  simp [axisInBounds?, h.proof]

/--
`NonemptyAxis axis s` says that `axis` selects a positive-length dimension of `s`.

The constructors follow the recursive representation of `Shape`, so reduction definitions can
eliminate this evidence while recursing through outer dimensions.
-/
inductive NonemptyAxis : Nat → Shape → Prop
  | zero {n : Nat} {s : Shape} : NonemptyAxis 0 (.dim (n + 1) s)
  | succ {n : Nat} {s : Shape} {axis : Nat} :
      NonemptyAxis axis s → NonemptyAxis (axis + 1) (.dim n s)

/-- A nonempty axis is, in particular, a valid axis. -/
theorem NonemptyAxis.toAxisInBounds {axis : Nat} {s : Shape} (h : NonemptyAxis axis s) :
    AxisInBounds axis s := by
  induction h with
  | zero => exact axisInBoundsZero
  | succ inner ih =>
      constructor
      simpa [rank, Nat.add_comm] using Nat.add_lt_add_right ih.proof 1

/-- Axis zero is nonempty when the outer dimension is positive. -/
@[simp] theorem nonemptyAxis_zero {n : Nat} {s : Shape} :
    NonemptyAxis 0 (.dim (n + 1) s) :=
  .zero

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
@[simp] theorem nonemptyAxis_succ {n : Nat} {s : Shape} {axis : Nat}
    (h : NonemptyAxis axis s) : NonemptyAxis (axis + 1) (.dim n s) :=
  .succ h

/--
Return evidence that `axis` addresses a positive dimension of `s`.

Executable consumers use this to recover the proposition required by typed tensor operations from
a raw runtime axis. Invalid axes, including axes into zero-sized dimensions, return `none`.
-/
def nonemptyAxis? (axis : Nat) : (s : Shape) → Option (PLift (NonemptyAxis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨NonemptyAxis.zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ inner, n =>
          (nonemptyAxis? inner rest).map (fun h =>
            ⟨NonemptyAxis.succ (n := n) (s := rest) (axis := inner) h.down⟩)

/-- Typeclass wrapper used when nonempty-axis evidence can be inferred statically. -/
class HasNonemptyAxis (axis : Nat) (s : Shape) : Prop where
  /-- The selected dimension has positive extent. -/
  proof : NonemptyAxis axis s

/-- Instance: axis `0` is valid for any positive outer dimension. -/
instance hasNonemptyAxisZero {n s} : HasNonemptyAxis 0 (.dim (n + 1) s) :=
  ⟨NonemptyAxis.zero⟩

/-- Package a proof that the outer dimension is nonzero as axis evidence. -/
theorem hasNonemptyAxisZeroOfNe {n s} (h : n ≠ 0) : HasNonemptyAxis 0 (.dim n s) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h
  ⟨by rw [hm]; exact .zero⟩

/-- Package positivity of the outer dimension as axis evidence. -/
theorem hasNonemptyAxisZeroOfPos {n s} (h : 0 < n) : HasNonemptyAxis 0 (.dim n s) :=
  hasNonemptyAxisZeroOfNe (Nat.ne_of_gt h)

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
instance hasNonemptyAxisSucc {n s axis} [h : HasNonemptyAxis axis s] :
    HasNonemptyAxis (axis + 1) (.dim n s) :=
  ⟨NonemptyAxis.succ h.proof⟩


/-!
## Well-formedness (`well_formed`)

`well_formed s` means "all dimensions are positive".

Why this matters (and why we designed it this way):

- Many definitions use `Fin n` indexing; if `n = 0`, there is no index and you end up with either
  vacuous truths or extra cases that obscure the intent of the lemma.
- Some common ops become awkward or partial at `n = 0`. For example, a mean typically divides by
  the number of elements, so `n = 0` needs special-case semantics.
- PyTorch *does* allow zero-sized dimensions, and most ops define a sensible result for them. We
  intentionally keep that complexity out of the core spec layer because it makes proofs much more
  case-heavy. When we need zero-dimension tensors, we introduce them with explicit
  semantics instead of relying on incidental behavior.

This is a pragmatic choice: proofs and specs are shorter, and
runtime checks can still handle edge cases separately.
-/
-- Well-formed shapes have positive dimensions.
/-- `well_formed s` means "all dimensions of `s` are positive" (recursively). -/
def wellFormed : Shape → Prop
| .scalar => True
| .dim n s => n > 0 ∧ s.wellFormed

/-!
### Size positivity

If all dimensions of a shape are positive, then the total number of scalar elements is positive.

This is a small but useful bridge lemma: many reductions are only defined for nonempty dimensions,
and `WellFormed` is our standard way of expressing that assumption.
-/

/-- If `s.well_formed`, then `Spec.Shape.size s > 0`. -/
theorem size_pos_of_well_formed : ∀ {s : Shape}, s.wellFormed → 0 < Spec.Shape.size s
  | .scalar, _ => by
      simp [Spec.Shape.size]
  | .dim n s, hw => by
      rcases hw with ⟨hn, hs⟩
      simpa [Spec.Shape.size] using Nat.mul_pos hn (size_pos_of_well_formed (s := s) hs)

/-- A shape of positive total size has no zero dimension. -/
theorem wellFormed_of_size_pos : ∀ {s : Shape}, 0 < Spec.Shape.size s → s.wellFormed
  | .scalar, _ => trivial
  | .dim n s, h => by
      have hn : n ≠ 0 := by
        intro hn
        simp [Spec.Shape.size, hn] at h
      have hs : Spec.Shape.size s ≠ 0 := by
        intro hs
        simp [Spec.Shape.size, hs] at h
      exact ⟨Nat.pos_of_ne_zero hn, wellFormed_of_size_pos (Nat.pos_of_ne_zero hs)⟩

/-- Every in-bounds axis of a well-formed shape has positive extent. -/
theorem nonemptyAxisOfWellFormed {s : Shape} (hw : s.wellFormed) {axis : Nat}
    (hAxis : axis < s.rank) : NonemptyAxis axis s := by
  induction s generalizing axis with
  | scalar => simp [rank] at hAxis
  | dim n rest ih =>
      rcases hw with ⟨hn, hRest⟩
      cases axis with
      | zero =>
          obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
          exact .zero
      | succ axis =>
          apply NonemptyAxis.succ
          apply ih hRest
          simp only [rank] at hAxis
          grind

/-- Package a valid axis of a well-formed shape as inferred reduction-axis evidence. -/
theorem hasNonemptyAxisOfWellFormed {s : Shape} (hw : s.wellFormed) {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  ⟨nonemptyAxisOfWellFormed hw hAxis⟩

/--
Typeclass wrapper for `Shape.well_formed`.

We use a typeclass (instead of passing a `well_formed` proof everywhere) because it mirrors how
other "side conditions" are handled in the library: call sites stay clean, and instances can be
provided locally (e.g. `letI : Shape.WellFormed s := ...`) when needed.
-/
class WellFormed (s : Shape) : Prop where
  proof : s.wellFormed

-- Scalars are always well-formed.
/-- Scalars are always well-formed. -/
instance : WellFormed .scalar where
  proof := trivial

/-- Adding a nonzero dimension preserves well-formedness. -/
instance {n s} [Shape.WellFormed s] [NeZero n] : Shape.WellFormed (.dim n s) :=
  ⟨⟨Nat.pos_of_ne_zero (NeZero.ne n), Shape.WellFormed.proof⟩⟩

/-- Infer nonemptiness of any valid axis from `WellFormed s`. -/
theorem inferNonemptyAxis {s : Shape} [hw : WellFormed s] {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  hasNonemptyAxisOfWellFormed hw.proof hAxis

/-!
`padLeft n s` prepends `n` singleton dimensions to a shape.

PyTorch analogy: `unsqueeze(0)` repeated `n` times (or equivalently viewing a tensor as having
extra leading dimensions of size 1). This is also the "prepend 1s" step you see in broadcasting.
-/
/-- Prepend `n` leading singleton dimensions (size `1`) to a shape. -/
def padLeft : Nat → Shape → Shape
| 0, s => s
| (n+1), s => dim 1 (padLeft n s)

-- Padding with leading `1`s increases rank by exactly `n`.
/-- `padLeft n s` increases the rank by exactly `n`. -/
theorem padLeft_rank : ∀ n s, (padLeft n s).rank = n + s.rank
| 0, s => by simp [padLeft]
| n+1, s => by
  simp [padLeft, rank]
  rw [padLeft_rank n]
  grind

end Shape
end Spec
