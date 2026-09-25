/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# Objective-Dependent Backward CROWN

Forward CROWN gives nodewise bounds. This module handles the complementary use case: start from a
linear objective on an output node and propagate that objective backward through the graph, choosing
local relaxations from the sign of the downstream coefficients.

For exact scalar backends, the coefficient transformations are ordinary algebraic identities. For
rounded scalar backends, TorchLean carries intervals for the coefficients and evaluates every
coefficient product and sum outwards through `BoundOps`. This is the executable enclosure
algorithm; its regression tests compare the result with directed IBP. An end-to-end theorem for the
rounded backward pass remains separate work, as does any claim relating a host runtime's evaluation
order to the reassociated backward expression.
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

/-!
The backward pass covers the same verifier dialect as `runCROWN` where objective-dependent
relaxations are available. Unsupported nodes consume already-computed IBP boxes conservatively.
-/

private inductive BackwardDir where
  | lower
  | upper

private structure BackwardState (α : Type) [Context α] where
  coeffs : Array (Option (FlatTensor α)) -- per-node objective coefficients
  cst    : α                         -- accumulated constant term
  failed : Bool := false              -- an active objective could not be propagated safely

private def BackwardState.fail (st : BackwardState α) : BackwardState α :=
  { st with failed := true }

private def flatTensorAdd (a b : FlatTensor α) : Option (FlatTensor α) :=
  if h : a.n = b.n then
    let bv : Tensor α [a.n] :=
      castDimScalar (α := α) (n := b.n) (n' := a.n) h.symm b.v
    some { n := a.n, v := Tensor.addSpec a.v bv }
  else
    none

private def flatTensorScale (k : α) (v : FlatTensor α) : FlatTensor α :=
  { n := v.n, v := Tensor.scaleSpec v.v k }

private def addCoeff (st : BackwardState α) (pid : Nat) (v : FlatTensor α) : BackwardState α :=
  match st.coeffs[pid]! with
  | none => { st with coeffs := st.coeffs.set! pid (some v) }
  | some w =>
    match flatTensorAdd (α:=α) w v with
    | some s => { st with coeffs := st.coeffs.set! pid (some s) }
    | none   => st.fail

private def dotFlat {n : Nat} (a b : Tensor α [n]) : α :=
  Spec.Tensor.sumSpec (Tensor.mulSpec a b)

private def consumeObjectiveFromBox (dir : BackwardDir) (aY : FlatTensor α) (B : FlatBox α) : Option α
  :=
  if h : aY.n = B.dim then
    let aYv : Tensor α [B.dim] :=
      castDimScalar (α := α) (n := aY.n) (n' := B.dim) h aY.v
    let fa := getDimScalarFn (α := α) aYv
    let flo := getDimScalarFn (α := α) B.lo
    let fhi := getDimScalarFn (α := α) B.hi
    let products : Array α :=
      (Array.finRange B.dim).map fun i =>
        match fa i, flo i, fhi i with
        | .scalar ay, .scalar l, .scalar u =>
          let y :=
            if decide (ay > Numbers.zero) then
              match dir with
              | .upper => u
              | .lower => l
            else
              match dir with
              | .upper => l
              | .lower => u
          match dir with
          | .lower => BoundOps.mulDown ay y
          | .upper => BoundOps.mulUp ay y
    some <| products.foldl
      (match dir with
      | .lower => BoundOps.addDown
      | .upper => BoundOps.addUp)
      Numbers.zero
  else
    none

/-- Add a constant contribution while preserving the direction of the requested enclosure. -/
private def addConstant (dir : BackwardDir) (st : BackwardState α) (c : α) : BackwardState α :=
  { st with cst :=
      match dir with
      | .lower => BoundOps.addDown st.cst c
      | .upper => BoundOps.addUp st.cst c }

/-- Regard a scalar objective enclosure as an affine form with zero input coefficients. -/
private def constantObjectiveAffine (inputDim : Nat) (c : α) : AffineVec α inputDim 1 :=
  { A := Spec.fill (α := α) Numbers.zero (.dim 1 (.dim inputDim .scalar))
    c := Tensor.dim (fun _ => Tensor.scalar c) }

/-- Bound an objective directly from the output box, without affine reassociation. -/
private def objectiveFromOutputBox
    (dir : BackwardDir) (ibp : Array (Option (FlatBox α)))
    (outputId inputDim : Nat) (obj : FlatTensor α) : Option (AffineVec α inputDim 1) := do
  let some outputBox := ibp[outputId]?
    | none
  let outputBox ← outputBox
  let bound ← consumeObjectiveFromBox (α := α) dir obj outputBox
  pure <| constantObjectiveAffine (α := α) inputDim bound

/-!
## Directed affine propagation

An interval coefficient `[a₋, a₊]` records all values that a mathematically exact backward
coefficient may take after the verifier has evaluated its arithmetic with outward rounding. Linear
and structural nodes preserve these coefficients. At a node without a directed affine rule, the
active objective is discharged against that node's directed IBP box.

The final conversion chooses one endpoint of each coefficient interval according to the sign of
the corresponding input interval. When an input interval crosses zero, a directed constant
correction accounts for the coefficient endpoint that was not selected.
-/

private structure DirectedBackwardState (α : Type) [Context α] where
  coeffs : Array (Option (FlatBox α))
  cstLo : α
  cstHi : α
  failed : Bool := false

private def DirectedBackwardState.fail
    (st : DirectedBackwardState α) : DirectedBackwardState α :=
  { st with failed := true }

private def pointCoeffBox (v : FlatTensor α) : FlatBox α :=
  { dim := v.n, lo := v.v, hi := v.v }

private def directedIntervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1Lo := BoundOps.mulDown aLo bLo
  let p2Lo := BoundOps.mulDown aLo bHi
  let p3Lo := BoundOps.mulDown aHi bLo
  let p4Lo := BoundOps.mulDown aHi bHi
  let p1Hi := BoundOps.mulUp aLo bLo
  let p2Hi := BoundOps.mulUp aLo bHi
  let p3Hi := BoundOps.mulUp aHi bLo
  let p4Hi := BoundOps.mulUp aHi bHi
  (min2 (min2 p1Lo p2Lo) (min2 p3Lo p4Lo),
    max2 (max2 p1Hi p2Hi) (max2 p3Hi p4Hi))

private def addDirectedCoeff
    (st : DirectedBackwardState α) (pid : Nat) (v : FlatBox α) :
    DirectedBackwardState α :=
  match st.coeffs[pid]! with
  | none => { st with coeffs := st.coeffs.set! pid (some v) }
  | some w =>
      if h : w.dim = v.dim then
        let vlo : Tensor α [w.dim] :=
          castDimScalar (α := α) (n := v.dim) (n' := w.dim) h.symm v.lo
        let vhi : Tensor α [w.dim] :=
          castDimScalar (α := α) (n := v.dim) (n' := w.dim) h.symm v.hi
        let sum : FlatBox α :=
          { dim := w.dim
            lo := Tensor.map2Spec BoundOps.addDown w.lo vlo
            hi := Tensor.map2Spec BoundOps.addUp w.hi vhi }
        { st with coeffs := st.coeffs.set! pid (some sum) }
      else
        st.fail

private def negateDirectedCoeff (v : FlatBox α) : FlatBox α :=
  { dim := v.dim
    lo := Tensor.mapSpec (fun x => BoundOps.subDown Numbers.zero x) v.hi
    hi := Tensor.mapSpec (fun x => BoundOps.subUp Numbers.zero x) v.lo }

private def directedDotBox (a b : FlatBox α) : Option (α × α) :=
  if h : a.dim = b.dim then
    let bLo : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.lo
    let bHi : Tensor α [a.dim] :=
      castDimScalar (α := α) (n := b.dim) (n' := a.dim) h.symm b.hi
    let terms := (List.finRange a.dim).map fun i =>
      directedIntervalMul (α := α)
        (getAtOrZero a.lo [i.val]) (getAtOrZero a.hi [i.val])
        (getAtOrZero bLo [i.val]) (getAtOrZero bHi [i.val])
    let lo := terms.foldl (fun acc p => BoundOps.addDown acc p.1) Numbers.zero
    let hi := terms.foldl (fun acc p => BoundOps.addUp acc p.2) Numbers.zero
    some (lo, hi)
  else
    none

private def addDirectedConstant
    (st : DirectedBackwardState α) (cLo cHi : α) : DirectedBackwardState α :=
  { st with
    cstLo := BoundOps.addDown st.cstLo cLo
    cstHi := BoundOps.addUp st.cstHi cHi }

private def consumeDirectedObjective
    (st : DirectedBackwardState α) (aY By : FlatBox α) : DirectedBackwardState α :=
  match directedDotBox (α := α) aY By with
  | some (lo, hi) => addDirectedConstant (α := α) st lo hi
  | none => st.fail

private def directedBackwardLinear {m n : Nat}
    (aY : FlatBox α) (W : Tensor α [m, n])
    (b : Tensor α [m]) : Option (FlatBox α × (α × α)) :=
  if h : aY.dim = m then
    let aLo : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.dim) (n' := m) h aY.lo
    let aHi : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.dim) (n' := m) h aY.hi
    let coeffAt (j : Fin n) : α × α :=
      (List.finRange m).foldl (fun acc i =>
        let w := getAtOrZero W [i.val, j.val]
        let p := directedIntervalMul (α := α)
          (getAtOrZero aLo [i.val]) (getAtOrZero aHi [i.val]) w w
        (BoundOps.addDown acc.1 p.1, BoundOps.addUp acc.2 p.2))
        (Numbers.zero, Numbers.zero)
    let aX : FlatBox α :=
      { dim := n
        lo := Tensor.dim (fun j => Tensor.scalar (coeffAt j).1)
        hi := Tensor.dim (fun j => Tensor.scalar (coeffAt j).2) }
    let bBox : FlatBox α := { dim := m, lo := b, hi := b }
    directedDotBox (α := α) { dim := m, lo := aLo, hi := aHi } bBox |>.map fun c => (aX, c)
  else
    none

private def splitDirectedCoeff
    (aY : FlatBox α) (n1 n2 : Nat) : Option (FlatBox α × FlatBox α) :=
  if h : aY.dim = n1 + n2 then
    let lo : Tensor α [n1 + n2] :=
      castDimScalar (α := α) (n := aY.dim) (n' := n1 + n2) h aY.lo
    let hi : Tensor α [n1 + n2] :=
      castDimScalar (α := α) (n := aY.dim) (n' := n1 + n2) h aY.hi
    let first : FlatBox α :=
      { dim := n1
        lo := Tensor.dim (fun i => Tensor.scalar (getAtOrZero lo [i.val]))
        hi := Tensor.dim (fun i => Tensor.scalar (getAtOrZero hi [i.val])) }
    let second : FlatBox α :=
      { dim := n2
        lo := Tensor.dim (fun i => Tensor.scalar (getAtOrZero lo [n1 + i.val]))
        hi := Tensor.dim (fun i => Tensor.scalar (getAtOrZero hi [n1 + i.val])) }
    some (first, second)
  else
    none

private def diagOfMat {n : Nat} (A : Tensor α [n, n]) : Tensor α [n] :=
  match A with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        match cols i with
        | .scalar v => Tensor.scalar v)

-- Apply a chosen diagonal relaxation y = s ⊙ x + b for a bound on a scalar objective.
private def backwardApplyDiag {n : Nat}
  (dir : BackwardDir)
  (aY : Tensor α [n])
  (sLo bLo sHi bHi : Tensor α [n]) :
  (Tensor α [n] × α) :=
  let fa := getDimScalarFn (α := α) aY
  let fsLo := getDimScalarFn (α := α) sLo
  let fbLo := getDimScalarFn (α := α) bLo
  let fsHi := getDimScalarFn (α := α) sHi
  let fbHi := getDimScalarFn (α := α) bHi
  let sChosen : Tensor α [n] :=
    Tensor.dim (fun i =>
      match fa i, fsLo i, fsHi i with
      | .scalar ay, .scalar slo, .scalar shi =>
        let s :=
          if decide (ay > Numbers.zero) then
            match dir with
            | .upper => shi
            | .lower => slo
          else
            match dir with
            | .upper => slo
            | .lower => shi
        Tensor.scalar s)
  let bChosen : Tensor α [n] :=
    Tensor.dim (fun i =>
      match fa i, fbLo i, fbHi i with
      | .scalar ay, .scalar blo, .scalar bhi =>
        let b :=
          if decide (ay > Numbers.zero) then
            match dir with
            | .upper => bhi
            | .lower => blo
          else
            match dir with
            | .upper => blo
            | .lower => bhi
        Tensor.scalar b)
  let aX := Tensor.mulSpec aY sChosen
  let cst := dotFlat (α:=α) aY bChosen
  (aX, cst)

-- Backward step for a unary op with diagonal relaxations
-- (relu/exp/log/sigmoid/tanh/softmax/layernorm).
private def backwardUnaryDiag
  (dir : BackwardDir) (preB : FlatBox α) (localB : FlatAffineBounds α)
  (aY : FlatTensor α) : Option (FlatTensor α × α) := by
  if h : aY.n = preB.dim then
    let n := preB.dim
    if hIn : localB.inDim = n then
      if hOut : localB.outDim = n then
        let aYv : Tensor α [n] :=
          castDimScalar (α := α) (n := aY.n) (n' := n) h aY.v
        let loAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.loAff)
        let hiAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.hiAff)
        let sLo := diagOfMat (α:=α) (n:=n) loAffN.A
        let bLo := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.loAff.c
        let sHi := diagOfMat (α:=α) (n:=n) hiAffN.A
        let bHi := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.hiAff.c
        let (aX, cst) := backwardApplyDiag (α:=α) (n:=n) dir aYv sLo bLo sHi bHi
        exact some ({ n := n, v := aX }, cst)
      else
        exact none
    else
      exact none
  else
    exact none

private def matLeftMul {m n : Nat}
  (aY : Tensor α [m]) (W : Tensor α [m, n]) :
  Tensor α [n] :=
  match aY, W with
  | .dim aF, .dim rows =>
    Tensor.materialize <|
      Tensor.dim (fun j =>
        let s : α :=
          (List.finRange m).foldl (fun acc i =>
            match aF i, rows i with
            | .scalar ai, .dim cols =>
              match cols j with
              | .scalar wij => acc + ai * wij) Numbers.zero
        Tensor.scalar s)
  | _, _ =>
    Spec.fill (α := α) Numbers.zero (.dim n .scalar)

private def backwardLinear {m n : Nat}
  (aY : FlatTensor α) (W : Tensor α [m, n]) (b : Tensor α [m]) :
  Option (FlatTensor α × α) :=
  if h : aY.n = m then
    let aYv : Tensor α [m] :=
      castDimScalar (α := α) (n := aY.n) (n' := m) h aY.v
    let aX := matLeftMul (α:=α) (m:=m) (n:=n) aYv W
    let cst := dotFlat (α:=α) aYv b
    some ({ n := n, v := aX }, cst)
  else
    none

private def backwardAdd (aY : FlatTensor α) : FlatTensor α := aY

private def backwardSubLeft (aY : FlatTensor α) : FlatTensor α := aY

private def backwardSubRight (aY : FlatTensor α) : FlatTensor α :=
  flatTensorScale (α:=α) (k := (-Numbers.one)) aY

private def backwardConcatSplit
  (aY : FlatTensor α) (n1 n2 : Nat) : Option (FlatTensor α × FlatTensor α) :=
  if h : aY.n = n1 + n2 then
    let aYv : Tensor α [n1 + n2] :=
      castDimScalar (α := α) (n := aY.n) (n' := n1 + n2) h aY.v
    let a1 : Tensor α [n1] :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [i.val]))
    let a2 : Tensor α [n2] :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [n1 + i.val]))
    some ({ n := n1, v := a1 }, { n := n2, v := a2 })
  else
    none

private def backwardPermuteVec {n : Nat} (perm : Fin n → Fin n) (v : Tensor α [n]) :
  Tensor α [n] :=
  match v with
  | .dim f => Tensor.dim (fun i => f (perm i))

/-- Pull a flat objective back through an arbitrary valid axis permutation. -/
private def backwardAxisPermutation? (outputShape : Shape) (forwardPerm : Array Nat)
    (aY : FlatTensor α) : Option (FlatTensor α) := do
  if aY.n = 0 then
    pure aY
  else
    let inverse ← (OpContracts.inversePerm forwardPerm).toOption
    let flatPerm ← flatAxisPermutation? outputShape inverse aY.n
    pure { n := aY.n, v := backwardPermuteVec (α := α) flatPerm aY.v }

private def backwardMatmul
  (dir : BackwardDir)
  (aZ : FlatTensor α) (Bx By : FlatBox α)
  (sA sB : Shape) :
  Option ((FlatTensor α) × (FlatTensor α) × α) :=
  let dims? : Option (Nat × Nat × Nat × Nat) :=
    match sA, sB with
    | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
      if k = k' then
        some (1, m, k, n)
      else
        none
    | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
      if hb : b = b' then
        match hb with
        | rfl =>
          if k = k' then
            some (b, m, k, n)
          else
            none
      else
        none
    | _, _ => none
  match dims? with
  | none => none
  | some (batch, m, k, n) =>
    let dimA := batch * m * k
    let dimB := batch * k * n
    let outDim := batch * m * n
    if aZ.n = outDim ∧ Bx.dim = dimA ∧ By.dim = dimB then
      let (aArr, bArr, cst) : Array α × Array α × α := Id.run do
        let mut aArr : Array α := Array.replicate dimA Numbers.zero
        let mut bArr : Array α := Array.replicate dimB Numbers.zero
        let mut cst : α := Numbers.zero
        let block : Nat := m * n
        let strideA : Nat := m * k
        let strideB : Nat := k * n
        for outIdx in List.range outDim do
          let az : α := getAtOrZero aZ.v [outIdx]
          let bi := outIdx / block
          let rem := outIdx % block
          let i := rem / n
          let j := rem % n
          let baseA := bi * strideA
          let baseB := bi * strideB
          for kk in List.range k do
            let aIdx := baseA + i * k + kk
            let bIdx := baseB + kk * n + j
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.half
            let cy := (ly + uy) * Numbers.half

            -- Upper plane selection.
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let axU := if u1 < u2 then ly else uy
            let ayU := if u1 < u2 then ux else lx
            let bU := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))

            -- Lower plane selection.
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let axL := if l1 > l2 then ly else uy
            let ayL := if l1 > l2 then lx else ux
            let bL := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))

            let useUpper : Bool :=
              if decide (az > Numbers.zero) then
                match dir with
                | .upper => true
                | .lower => false
              else
                match dir with
                | .upper => false
                | .lower => true

            let ax := if useUpper then axU else axL
            let ay := if useUpper then ayU else ayL
            let bb := if useUpper then bU else bL

            aArr := aArr.set! aIdx (aArr[aIdx]! + az * ax)
            bArr := bArr.set! bIdx (bArr[bIdx]! + az * ay)
            cst := cst + az * bb
        return (aArr, bArr, cst)

      let aT : Tensor α [dimA] :=
        Tensor.dim (fun i => Tensor.scalar (aArr[i.val]!))
      let bT : Tensor α [dimB] :=
        Tensor.dim (fun i => Tensor.scalar (bArr[i.val]!))
      some ({ n := dimA, v := aT }, { n := dimB, v := bT }, cst)
    else
      none

private def backwardMulElem
  (dir : BackwardDir)
  (aZ : FlatTensor α) (Bx By : FlatBox α) :
  Option ((FlatTensor α) × (FlatTensor α) × α) :=
  if h : aZ.n = Bx.dim ∧ Bx.dim = By.dim then
    let n := Bx.dim
    let hZ : aZ.n = n := h.1
    let aZv : Tensor α [n] :=
      castDimScalar (α := α) (n := aZ.n) (n' := n) hZ aZ.v
    let xLo := getDimScalarFn (α := α) Bx.lo
    let xHi := getDimScalarFn (α := α) Bx.hi
    let yLo := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.lo)
    let yHi := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.hi)
    let aF := getDimScalarFn (α := α) aZv
    -- Choose one McCormick plane per element using the interval midpoint.
    let axU : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ax := if u1 < u2 then ly else uy
          Tensor.scalar ax)
    let ayU : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ay := if u1 < u2 then ux else lx
          Tensor.scalar ay)
    let bU : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let b := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
          Tensor.scalar b)
    let axL : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ax := if l1 > l2 then ly else uy
          Tensor.scalar ax)
    let ayL : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ay := if l1 > l2 then lx else ux
          Tensor.scalar ay)
    let bL : Tensor α [n] :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.half
          let my := (ly + uy) * Numbers.half
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let b := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
          Tensor.scalar b)
    let axUFn := getDimScalarFn (α := α) axU
    let ayUFn := getDimScalarFn (α := α) ayU
    let bUFn := getDimScalarFn (α := α) bU
    let axLFn := getDimScalarFn (α := α) axL
    let ayLFn := getDimScalarFn (α := α) ayL
    let bLFn := getDimScalarFn (α := α) bL
    let aX : Tensor α [n] :=
      Tensor.dim (fun i =>
        match aF i, axUFn i, axLFn i with
        | .scalar az, .scalar axu, .scalar axl =>
          let ax :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => axu
              | .lower => axl
            else
              match dir with
              | .upper => axl
              | .lower => axu
          Tensor.scalar (az * ax))
    let aY : Tensor α [n] :=
      Tensor.dim (fun i =>
        match aF i, ayUFn i, ayLFn i with
        | .scalar az, .scalar ayu, .scalar ayl =>
          let ay :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => ayu
              | .lower => ayl
            else
              match dir with
              | .upper => ayl
              | .lower => ayu
          Tensor.scalar (az * ay))
    let biasProd : Tensor α [n] :=
      Tensor.dim (fun i =>
        match aF i, bUFn i, bLFn i with
        | .scalar az, .scalar bu, .scalar bl =>
          let b :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => bu
              | .lower => bl
            else
              match dir with
              | .upper => bl
              | .lower => bu
          Tensor.scalar (az * b))
    let cst := Spec.Tensor.sumSpec biasProd
    some ({ n := n, v := aX }, { n := n, v := aY }, cst)
  else
    none

private def backwardNode (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .input =>
      if node.id = ctx.inputId then
        st
      else
        match ibp[id]! with
        | some Bx =>
          match consumeObjectiveFromBox (α := α) (dir := dir) aY Bx with
          | some cadd => addConstant ( α := α) dir st cadd
          | none => st.fail
        | none => st.fail
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        if h : aY.n = v.n then
          let aYv : Tensor α [v.n] :=
            castDimScalar (α := α) (n := aY.n) (n' := v.n) h aY.v
          let add := dotFlat (α:=α) aYv v.v
          addConstant (α := α) dir st add
        else st.fail
      | none => st.fail
    | .detach =>
      match node.parents with
      | #[p1] => addCoeff (α := α) st p1 aY
      | _ => st.fail
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        let st1 := addCoeff (α:=α) st p1 (backwardAdd (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardAdd (α:=α) aY)
      | _ => st.fail
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        let st1 := addCoeff (α:=α) st p1 (backwardSubLeft (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardSubRight (α:=α) aY)
      | _ => st.fail
    | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .sin | .cos | .maxElem |
      .minElem | .hardMaskedSoftmax _
    | .maxPool .. | .avgPool ..
    | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail
    | .batchNormEval channelAxis _ =>
      match node.parents with
      | #[p1] =>
        match ps.batchNormEval[id]? with
        | some cfg =>
          match batchNormEvalLinear? (α := α) nodes[p1]!.outShape channelAxis cfg with
          | some p =>
            match backwardLinear (α := α) (m := p.m) (n := p.n) aY p.w p.b with
            | some (aX, cadd) =>
              let st' := addCoeff (α := α) st p1 aX
              addConstant (α := α) dir st' cadd
            | none => st.fail
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .linear =>
      match node.parents with
      | #[p1] =>
        match ps.linearWB[id]? with
        | some p =>
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w p.b with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            addConstant (α := α) dir st' cadd
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .matmul =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMatmul (α:=α) (dir:=dir) aY Bx By (sA := nodes[p1]!.outShape) (sB :=
            nodes[p2]!.outShape) with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            addConstant (α := α) dir st2 cadd
          | none => st.fail
        | _, _ => st.fail
      | #[p1] =>
        match ps.matmulW[id]? with
        | some p =>
          let zb := Spec.fill (α := α) Numbers.zero (.dim p.m .scalar)
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w zb with
          | some (aX, _cadd) =>
            addCoeff (α:=α) st p1 aX
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | .conv .. =>
      if !crownNodeSemanticsSupported (α := α) nodes ps id then
        st.fail
      else
        match node.parents with
        | #[p1] =>
          match ps.convCfg[id]? with
          | some cfg =>
            let inShape := Shape.ofList (cfg.inChannels :: cfg.inputSpatial.toList)
            let outSpatial := Spec.convOutSpatial cfg.inputSpatial cfg.kernel cfg.stride cfg.padding
            let outShape := Shape.ofList (cfg.outChannels :: outSpatial.toList)
            let convAff := affOfConv (α:=α) cfg
            match backwardLinear (α:=α) (m:=outShape.size) (n:=inShape.size) aY convAff.A
              convAff.c with
            | some (aX, cadd) =>
              let st' := addCoeff (α:=α) st p1 aX
              addConstant (α := α) dir st' cadd
            | none => st.fail
          | none => st.fail
        | _ => st.fail
    | .layernorm _ =>
      if !crownNodeSemanticsSupported (α := α) nodes ps id then
        st.fail
      else
        match ibp[id]! with
        | some By =>
          match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
          | some cadd => addConstant (α := α) dir st cadd
          | none => st.fail
        | none => st.fail
    -- MinMax: fail rather than pass anything through.  The value pass produces no box for it
    -- (`FlatBox` has lost the channel axis the pairing is defined over), so there is nothing to
    -- consume; `st.fail` is this engine's way of saying no objective bound could be derived.
    | .minMax _ => st.fail
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ =>
      -- The value pass has already applied the scalar backend's directed nonlinear capabilities.
      -- The default backward pass consumes that box rather than rebuilding a relaxation with
      -- exact-real algebra. The alpha-specific entry point below retains its explicit ReLU rule.
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail
    | .mul_elem =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMulElem (α:=α) (dir:=dir) aY Bx By with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            addConstant (α := α) dir st2 cadd
          | none => st.fail
        | _, _ => st.fail
      | _ => st.fail
    | .sum =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]! with
        | some Bx =>
          if aY.n = 1 then
            let a0 : α := getAtOrZero aY.v [0]
            let out : FlatTensor α :=
              { n := Bx.dim, v := Spec.fill (α := α) a0 (.dim Bx.dim .scalar) }
            addCoeff (α:=α) st p1 out
          else st.fail
        | none => st.fail
      | _ => st.fail
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | #[p1] => addCoeff (α:=α) st p1 aY
      | _ => st.fail
    | .concat axis =>
      if axis != 0 then
        st.fail
      else
        match node.parents with
        | #[p1, p2] =>
          match ibp[p1]!, ibp[p2]! with
          | some B1, some B2 =>
            match backwardConcatSplit (α:=α) aY B1.dim B2.dim with
            | some (a1, a2) =>
              let st1 := addCoeff (α:=α) st p1 a1
              addCoeff (α:=α) st1 p2 a2
            | none => st.fail
          | _, _ => st.fail
        | _ => st.fail
    | .transpose axis₁ axis₂ =>
      match node.parents with
      | #[p1] =>
        match OpContracts.transposePerm nodes[p1]!.outShape.rank axis₁ axis₂ with
        | .ok perm =>
          match backwardAxisPermutation? (α := α) node.outShape perm aY with
          | some aX => addCoeff (α := α) st p1 aX
          | none => st.fail
        | .error _ => st.fail
      | _ => st.fail
    | .permute perm =>
      match node.parents with
      | #[p1] =>
        match backwardAxisPermutation? (α := α) node.outShape perm aY with
        | some aX => addCoeff (α := α) st p1 aX
        | none => st.fail
      | _ => st.fail
    | .mseLoss =>
      -- The directed IBP pass already encloses the rounded subtraction, square, and mean. Reusing
      -- that enclosure avoids introducing an unqualified finite-precision quadratic relaxation.
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => addConstant (α := α) dir st cadd
        | none => st.fail
      | none => st.fail

private def backwardNodeWithReluAlpha (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (reluAlpha : Array (Option (FlatTensor α)))
  (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .relu =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB? : Option (FlatAffineBounds α) :=
            match reluAlpha[id]? with
            | some (some a) =>
              if h : a.n = n then
                let aT : Tensor α [n] :=
                  castDimScalar (α:=α) (n:=a.n) (n':=n) h a.v
                some (propagateReluBoundsWithAlpha (α:=α) preB idB rfl aT)
              else
                some (propagateReluBounds (α:=α) preB idB rfl)
            | _ =>
              some (propagateReluBounds (α:=α) preB idB rfl)
          match localB? with
          | some localB =>
              match backwardUnaryDiag (α:=α) dir preB localB aY with
              | some (aX, cadd) =>
                let st' := addCoeff (α:=α) st p1 aX
                addConstant (α := α) dir st' cadd
              | none => st.fail
          | none => st.fail
        | none => st.fail
      | _ => st.fail
    | _ =>
      backwardNode (α:=α) dir nodes ps ibp ctx st id

private def runBackwardObjectiveDir
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNode (α:=α) dir g.nodes ps ibp ctx acc i) init
    if st.failed then
      none
    else
      match st.coeffs[ctx.inputId]! with
      | some aIn =>
        if hIn : aIn.n = ctx.inputDim then
          let vIn : Tensor α [ctx.inputDim] :=
            castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
          let A : Tensor α [1, ctx.inputDim] := Tensor.dim (fun _ => vIn)
          let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
          some { A := A, c := c }
        else
          none
      | none =>
        -- Every active coefficient was consumed by input-independent nodes.
        let A : Tensor α [1, ctx.inputDim] :=
          Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
        let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
  else
    none

private def runBackwardObjectiveDirWithReluAlpha
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α)
  (reluAlpha : Array (Option (FlatTensor α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNodeWithReluAlpha (α:=α) dir g.nodes ps ibp ctx reluAlpha acc i) init
    if st.failed then
      none
    else
      match st.coeffs[ctx.inputId]! with
      | some aIn =>
        if hIn : aIn.n = ctx.inputDim then
          let vIn : Tensor α [ctx.inputDim] :=
            castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
          let A : Tensor α [1, ctx.inputDim] := Tensor.dim (fun _ => vIn)
          let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
          some { A := A, c := c }
        else
          none
      | none =>
        let A : Tensor α [1, ctx.inputDim] :=
          Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
        let c : Tensor α [1] := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
  else
    none

private def directedBackwardNode
    (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (ctx : AffineCtx) (st : DirectedBackwardState α) (id : Nat) :
    DirectedBackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
      let node := nodes[id]!
      let consumeCurrent :=
        match ibp[id]! with
        | some By => consumeDirectedObjective (α := α) st aY By
        | none => st.fail
      match node.kind with
      | .input =>
          if node.id = ctx.inputId then
            st
          else
            consumeCurrent
      | .const _ =>
          match ps.constVals[id]? with
          | some v => consumeDirectedObjective (α := α) st aY (pointCoeffBox (α := α) v)
          | none => st.fail
      | .detach =>
          match node.parents with
          | #[p] => addDirectedCoeff (α := α) st p aY
          | _ => st.fail
      | .add =>
          match node.parents with
          | #[p1, p2] =>
              let st := addDirectedCoeff (α := α) st p1 aY
              addDirectedCoeff (α := α) st p2 aY
          | _ => st.fail
      | .sub =>
          match node.parents with
          | #[p1, p2] =>
              let st := addDirectedCoeff (α := α) st p1 aY
              addDirectedCoeff (α := α) st p2 (negateDirectedCoeff (α := α) aY)
          | _ => st.fail
      | .linear =>
          match node.parents with
          | #[p] =>
              match ps.linearWB[id]? with
              | some cfg =>
                  match directedBackwardLinear (α := α) aY cfg.w cfg.b with
                  | some (aX, c) =>
                      let st := addDirectedCoeff (α := α) st p aX
                      addDirectedConstant (α := α) st c.1 c.2
                  | none => st.fail
              | none => st.fail
          | _ => st.fail
      | .matmul =>
          match node.parents with
          | #[p] =>
              match ps.matmulW[id]? with
              | some cfg =>
                  let zeroBias := Spec.fill (α := α) Numbers.zero (.dim cfg.m .scalar)
                  match directedBackwardLinear (α := α) aY cfg.w zeroBias with
                  | some (aX, c) =>
                      let st := addDirectedCoeff (α := α) st p aX
                      addDirectedConstant (α := α) st c.1 c.2
                  | none => st.fail
              | none => st.fail
          | _ => consumeCurrent
      | .conv .. =>
          if !crownNodeSemanticsSupported (α := α) nodes ps id then
            st.fail
          else
            match node.parents with
            | #[p] =>
                match ps.convCfg[id]? with
                | some cfg =>
                    let inShape := Shape.ofList (cfg.inChannels :: cfg.inputSpatial.toList)
                    let outSpatial := Spec.convOutSpatial cfg.inputSpatial cfg.kernel cfg.stride
                      cfg.padding
                    let outShape := Shape.ofList (cfg.outChannels :: outSpatial.toList)
                    let aff := affOfConv (α := α) cfg
                    match directedBackwardLinear (α := α)
                        (m := outShape.size) (n := inShape.size) aY aff.A aff.c with
                    | some (aX, c) =>
                        let st := addDirectedCoeff (α := α) st p aX
                        addDirectedConstant (α := α) st c.1 c.2
                    | none => st.fail
                | none => st.fail
            | _ => st.fail
      | .sum =>
          match node.parents with
          | #[p] =>
              match ibp[p]! with
              | some Bx =>
                  if h : aY.dim = 1 then
                    let lo : Tensor α [1] :=
                      castDimScalar (α := α) (n := aY.dim) (n' := 1) h aY.lo
                    let hi : Tensor α [1] :=
                      castDimScalar (α := α) (n := aY.dim) (n' := 1) h aY.hi
                    let aX : FlatBox α :=
                      { dim := Bx.dim
                        lo := Spec.fill (α := α) (getAtOrZero lo [0]) (.dim Bx.dim .scalar)
                        hi := Spec.fill (α := α) (getAtOrZero hi [0]) (.dim Bx.dim .scalar) }
                    addDirectedCoeff (α := α) st p aX
                  else
                    st.fail
              | none => st.fail
          | _ => st.fail
      | .reshape _ _ | .flatten _ =>
          match node.parents with
          | #[p] => addDirectedCoeff (α := α) st p aY
          | _ => st.fail
      | .concat axis =>
          if axis != 0 then
            st.fail
          else
            match node.parents with
            | #[p1, p2] =>
                match ibp[p1]!, ibp[p2]! with
                | some B1, some B2 =>
                    match splitDirectedCoeff (α := α) aY B1.dim B2.dim with
                    | some (a1, a2) =>
                        let st := addDirectedCoeff (α := α) st p1 a1
                        addDirectedCoeff (α := α) st p2 a2
                    | none => st.fail
                | _, _ => st.fail
            | _ => st.fail
      | .layernorm _ =>
          if !crownNodeSemanticsSupported (α := α) nodes ps id then st.fail else consumeCurrent
      | _ => consumeCurrent

private def directedInputAffines
    (inputDim : Nat) (xB aIn : FlatBox α) (cLo cHi : α) :
    Option (AffineVec α inputDim 1 × AffineVec α inputDim 1) :=
  if hx : xB.dim = inputDim then
    if ha : aIn.dim = inputDim then
      let xLo : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := xB.dim) (n' := inputDim) hx xB.lo
      let xHi : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := xB.dim) (n' := inputDim) hx xB.hi
      let aLo : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := aIn.dim) (n' := inputDim) ha aIn.lo
      let aHi : Tensor α [inputDim] :=
        castDimScalar (α := α) (n := aIn.dim) (n' := inputDim) ha aIn.hi
      let selected (i : Fin inputDim) : (α × α) × (α × α) :=
        let l := getAtOrZero xLo [i.val]
        let u := getAtOrZero xHi [i.val]
        let al := getAtOrZero aLo [i.val]
        let au := getAtOrZero aHi [i.val]
        if decide (¬ l < Numbers.zero) then
          ((al, Numbers.zero), (au, Numbers.zero))
        else if decide (¬ Numbers.zero < u) then
          ((au, Numbers.zero), (al, Numbers.zero))
        else
          let lowerCorrection :=
            BoundOps.mulDown (BoundOps.subUp au al) l
          let upperCorrection :=
            BoundOps.mulUp (BoundOps.subDown al au) l
          ((al, lowerCorrection), (au, upperCorrection))
      let lowerRow : Tensor α [inputDim] :=
        Tensor.dim (fun i => Tensor.scalar (selected i).1.1)
      let upperRow : Tensor α [inputDim] :=
        Tensor.dim (fun i => Tensor.scalar (selected i).2.1)
      let lowerCorrection := (List.finRange inputDim).foldl
        (fun acc i => BoundOps.addDown acc (selected i).1.2) Numbers.zero
      let upperCorrection := (List.finRange inputDim).foldl
        (fun acc i => BoundOps.addUp acc (selected i).2.2) Numbers.zero
      let lowerConstant := BoundOps.addDown cLo lowerCorrection
      let upperConstant := BoundOps.addUp cHi upperCorrection
      let lower : AffineVec α inputDim 1 :=
        { A := Tensor.dim (fun _ => lowerRow)
          c := Tensor.dim (fun _ => Tensor.scalar lowerConstant) }
      let upper : AffineVec α inputDim 1 :=
        { A := Tensor.dim (fun _ => upperRow)
          c := Tensor.dim (fun _ => Tensor.scalar upperConstant) }
      some (lower, upper)
    else
      none
  else
    none

private def runDirectedBackwardObjective
    (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
    Option (AffineVec α ctx.inputDim 1 × AffineVec α ctx.inputDim 1) := do
  if outputId < g.nodes.size then
    let initCoeffs :=
      (Array.replicate g.nodes.size none).set! outputId (some (pointCoeffBox (α := α) obj))
    let init : DirectedBackwardState α :=
      { coeffs := initCoeffs, cstLo := Numbers.zero, cstHi := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl
      (fun acc i => directedBackwardNode (α := α) g.nodes ps ibp ctx acc i) init
    if st.failed then
      none
    else
      let inputBox ← ibp[ctx.inputId]?
      let inputBox ← inputBox
      let aIn := st.coeffs[ctx.inputId]!.getD
        { dim := ctx.inputDim
          lo := Spec.fill (α := α) Numbers.zero (.dim ctx.inputDim .scalar)
          hi := Spec.fill (α := α) Numbers.zero (.dim ctx.inputDim .scalar) }
      directedInputAffines (α := α) ctx.inputDim inputBox aIn st.cstLo st.cstHi
  else
    none

/--
Objective-dependent backward CROWN bound for a scalar objective.

Given a linear objective `objᵀ * output`, this runs a backward pass that propagates the objective
coefficients through the graph, selects the relaxation attached to each node, and returns a pair of
affine bounds on the objective with respect to `ctx.inputId`.

The returned `FlatAffineBounds` always has `outDim = 1` (a scalar objective).
-/
def runCROWNBackwardObjective
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α) :
  Option (FlatAffineBounds α) :=
  if crownGraphSemanticsSupported (α := α) g ps then
    let bounds :=
      if BoundOps.supportsExactAffineReassociation (α := α) then
        (runBackwardObjectiveDir (α := α) .lower g ps ctx ibp outputId obj,
          runBackwardObjectiveDir (α := α) .upper g ps ctx ibp outputId obj)
      else
        match runDirectedBackwardObjective (α := α) g ps ctx ibp outputId obj with
        | some bounds => (some bounds.1, some bounds.2)
        | none =>
            (objectiveFromOutputBox (α := α) .lower ibp outputId ctx.inputDim obj,
              objectiveFromOutputBox (α := α) .upper ibp outputId ctx.inputDim obj)
    match bounds with
    | (some loAff, some hiAff) =>
        some { inDim := ctx.inputDim, outDim := 1, loAff := loAff, hiAff := hiAff }
    | _ => none
  else
    none

/-- Evaluate already-computed backward-CROWN objective bounds on an input box. -/
def evalBackwardObjectiveBox? (bounds : FlatAffineBounds α) (xB : FlatBox α)
    (inputDim : Nat) : Except String (FlatBox α) := do
  if hIn : bounds.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      if hOut : bounds.outDim = 1 then
        let outB := bounds.evalOnFlatBoxAsDim xB (by simpa [hXB] using hIn.symm) hOut
        pure { dim := 1, lo := outB.lo, hi := outB.hi }
      else
        throw s!"backward CROWN objective dimension mismatch: got {bounds.outDim}, expected 1"
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"backward CROWN input dimension mismatch: got {bounds.inDim}, expected {inputDim}"

/--
Run objective-dependent backward CROWN and evaluate the scalar objective bounds on the input box.

The result is a `FlatBox` of dimension `1`, with `lo[0]` and `hi[0]` bounding
`objᵀ * output` over `xB`.
-/
def backwardObjectiveBox? (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (xB : FlatBox α)
    (outputId : Nat) (obj : FlatTensor α) : Except String (FlatBox α) := do
  let some bounds := runCROWNBackwardObjective (α := α) g ps ctx ibp outputId obj
    | throw "CROWN backward objective failed"
  evalBackwardObjectiveBox? (α := α) bounds xB ctx.inputDim

/--
Backward CROWN objective lower bound with externally provided ReLU alpha slopes.

This is an integration hook for alpha-CROWN style workflows where ReLU slopes are optimized outside
TorchLean and then imported as a per-node vector in `reluAlpha`. Imported slopes currently refine
the exact scalar path. Rounded scalar backends use the directed coefficient pass; until imported
slopes carry their own rounding contract, nonlinear nodes are discharged against directed IBP
boxes.
-/
def runCROWNBackwardObjectiveLowerWithReluAlpha
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatTensor α)
  (reluAlpha : Array (Option (FlatTensor α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if !crownGraphSemanticsSupported (α := α) g ps then
    none
  else if BoundOps.supportsExactAffineReassociation (α := α) then
    runBackwardObjectiveDirWithReluAlpha (α := α) .lower g ps ctx ibp outputId obj reluAlpha
  else
    (runDirectedBackwardObjective (α := α) g ps ctx ibp outputId obj).map (fun bounds => bounds.1)

end NN.MLTheory.CROWN.Graph
