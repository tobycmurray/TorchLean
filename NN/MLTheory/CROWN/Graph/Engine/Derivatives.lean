/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP

/-!
# Derivative Interval Passes

These passes propagate interval bounds for first and second derivatives through the same flat graph
used by IBP. Derivative propagation has its own chain-rule state but reuses `FlatBox` for every
intermediate enclosure.

Linear operations, pointwise arithmetic, supported activations, and selected structural operations
have explicit rules. Coupled softmax and layer-normalization derivatives are evaluated only when
the scalar instance declares their algebra exact. Finite-precision instances otherwise leave those
nodes unresolved instead of running the real-arithmetic formulas with rounded operations.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]
variable [NonlinearBoundOps α]

open BoundOps

/-- Global enclosure for the derivative of `tanh`. -/
private def tanhDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
    hi := Spec.fill (α := α) Numbers.one (.dim dim .scalar) }

/-- Global enclosure for the derivative of the logistic sigmoid. -/
private def sigmoidDerivBox (dim : Nat) : FlatBox α :=
  let quarter := BoundOps.mulUp Numbers.half Numbers.half
  { dim := dim
    lo := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
    hi := Spec.fill (α := α) quarter (.dim dim .scalar) }

/-- A simple global enclosure for the second derivative of `tanh`. -/
private def tanhSecondDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Spec.fill (α := α) (-Numbers.two) (.dim dim .scalar)
    hi := Spec.fill (α := α) Numbers.two (.dim dim .scalar) }

/-- A simple global enclosure for the second derivative of the logistic sigmoid. -/
private def sigmoidSecondDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Spec.fill (α := α) Numbers.negOne (.dim dim .scalar)
    hi := Spec.fill (α := α) Numbers.one (.dim dim .scalar) }

/-- Shared first-derivative propagation with a caller-supplied input seed. -/
private def runFirstDerivativeWithSeed
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (inputSeed : FlatBox α → Option (FlatBox α)) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        match inputSeed B with
        | some seed => drs.set! id (some seed)
        | none => drs
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool .. | .avgPool .. =>
      -- Not supported by the derivative-bound passes (used by PINN tooling).
      drs
    | .hardMaskedSoftmax _ =>
      -- The current derivative pass treats softmax as one flat vector. A masked attention tensor
      -- is row structured, so propagating that rule here would mix independent rows.
      drs
    | .sum =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dXin => drs.set! id (some (boxSum (α := α) dXin))
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    -- MinMax: no derivative enclosure is produced.  Its derivative is a data-dependent selection
    -- (which of a pair is the minimum), and this engine works on `FlatBox`, which has lost the
    -- channel axis the pairing is defined over.  Leaving the entry unset is this function's
    -- convention for a node it cannot differentiate: no bound is asserted, rather than a wrong one.
    | .minMax _ => drs
    | .relu =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dIn =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim dIn.dim .scalar)
          let o := Spec.fill (α:=α) Numbers.one  (.dim dIn.dim .scalar)
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match boxMulElem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dZ =>
          match boxMulElem (α := α) dZ (tanhDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dZ =>
          match boxMulElem (α := α) dZ (sigmoidDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .softmax _ =>
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then drs else
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          -- The formulas below construct one dense softmax Jacobian. They are sound only when
          -- the node itself is a vector, not when several last-axis rows share one flat box.
          if h : dZ.dim = yB.dim ∧ node.outShape = .dim yB.dim .scalar then
            let n := yB.dim
            -- Cast derivative tensors to dimension n for Fin alignment
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h.1) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h.1) dZ.hi
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let fdLo := getDimScalarFn (α:=α) dLo
            let fdHi := getDimScalarFn (α:=α) dHi
            let mulI (aLo aHi bLo bHi : α) : α × α :=
              let p1 := aLo * bLo; let p2 := aLo * bHi
              let p3 := aHi * bLo; let p4 := aHi * bHi
              let lo1 := if p1 < p2 then p1 else p2
              let lo2 := if p3 < p4 then p3 else p4
              let lo  := if lo1 < lo2 then lo1 else lo2
              let hi1 := if p1 > p2 then p1 else p2
              let hi2 := if p3 > p4 then p3 else p4
              let hi  := if hi1 > hi2 then hi1 else hi2
              (lo, hi)
            let dlo :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumHi)
            drs.set! id (some { dim := n, lo := dlo, hi := dhi })
          else drs
        | _, _ => drs
      | _ => drs
    | .sin =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some dF =>
            match boxMulElem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB with
          | some sB =>
            match boxMulElem (α:=α) dZ (boxNeg (α := α) sB) with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match derivBoxExp? (α := α) zB with
          | some dF =>
            match chainMul (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match derivBoxLog? (α := α) zB with
          | some dF =>
            match chainMul (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (boxAdd (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (boxSub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mul_elem =>
      match node.parents with
      | #[p1, p2] =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match boxMulElem (α:=α) dx yB, boxMulElem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (boxAdd (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      -- Derivative of layernorm y = (x - mean(x))/sqrt(var+eps): dy ≈ t*(dx - mean dx) + dt*u.
      -- We bound t in [t_lo,t_hi], bound v := (dx - mean dx) per-component, and bound |dt| via
      -- |dt| ≤ 0.5 * (var+eps)^(-3/2)_hi * (2/n) * Σ_j max|u_j| * max|v_j|; then add symmetric dt*u
      -- term.
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then drs else
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
        | some dXin, some Xin =>
          let n := Xin.dim
          if hn : dXin.dim = n then
            -- Compute mean bounds of x and dx
            let muBounds := idealLayerNormMeanBounds (α := α) Xin.lo Xin.hi
            -- Empty vectors have no coordinates to certify; use denominator 1 only to
            -- avoid evaluating `1 / 0` in the vacuous branch.
            let nDen : Nat := if n = 0 then 1 else n
            let nA : α := (nDen : Nat)
            let mu_lo := muBounds.1
            let mu_hi := muBounds.2
            -- u_j bounds = x_j - mean(x)
            let uBounds := idealLayerNormCenteredBounds (α := α) Xin.lo Xin.hi mu_lo mu_hi
            let u_lo := uBounds.1
            let u_hi := uBounds.2
            -- Bounds on variance and denom s = sqrt(var+eps)
            let var_hi := idealLayerNormVarianceUpper (α := α) Xin.lo Xin.hi mu_lo mu_hi
            let s_lo := MathFunctions.sqrt Numbers.epsilon
            let tBounds := idealLayerNormInvStdBounds (α := α) var_hi
            let t_lo := tBounds.1
            let t_hi := tBounds.2
            -- dx mean bounds and v_j = dx_j - mean(dx)
            let dmuBounds := idealLayerNormMeanBounds (α := α) dXin.lo dXin.hi
            let vBounds := idealLayerNormCenteredBounds (α := α) dXin.lo dXin.hi dmuBounds.1 dmuBounds.2
            let v_lo := vBounds.1
            let v_hi := vBounds.2
            -- Align v bounds to dimension n via cast
            let v_loN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_lo
            let v_hiN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_hi
            -- First term: t * v
            -- Compute base = t * v per component where t∈[t_lo,t_hi] and v_i∈[v_loN[i],v_hiN[i]]
            let vLoFn := getDimScalarFn (α:=α) v_loN
            let vHiFn := getDimScalarFn (α:=α) v_hiN
            let base_lo :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let m1 := if p1 < p2 then p1 else p2
                  let m2 := if p3 < p4 then p3 else p4
                  Tensor.scalar (if m1 < m2 then m1 else m2))
            let base_hi :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let M1 := if p1 > p2 then p1 else p2
                  let M2 := if p3 > p4 then p3 else p4
                  Tensor.scalar (if M1 > M2 then M1 else M2))
            let baseN : FlatBox α := { dim := n, lo := base_lo, hi := base_hi }
            -- Bound |dt| using t3_hi = 1/s^3 and |(2/n) Σ u_j v_j|
            let t3_hi :=
              let s_lo' := s_lo
              let s3 := s_lo' * s_lo' * s_lo'
              Numbers.one / (if s3 > Numbers.epsilon then s3 else Numbers.epsilon)
            let abs_max (l u : α) : α :=
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              if al > au then al else au
            -- compute G = Σ max|u_j| * max|v_j|
            let u_abs := getDimScalarFn (α:=α) u_lo
            let u_abs_hi := getDimScalarFn (α:=α) u_hi
            let v_abs := getDimScalarFn (α:=α) v_loN
            let v_abs_hi := getDimScalarFn (α:=α) v_hiN
            let G : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
              match u_abs i, u_abs_hi i, v_abs i, v_abs_hi i with
              | .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                let au := abs_max ul uu
                let av := abs_max vl vu
                acc + (au * av)
            ) 0
            let V := (Numbers.two * G) / nA
            let dt_abs := (Numbers.half * t3_hi) * V
            -- Add symmetric dt*u term per component: ± dt_abs * max|u_i|
            let fulo := getDimScalarFn (α:=α) u_lo
            let fuhi := getDimScalarFn (α:=α) u_hi
            let bLoFn := getDimScalarFn (α:=α) baseN.lo
            let bHiFn := getDimScalarFn (α:=α) baseN.hi
            let add_lo :=
              Tensor.dim (fun i =>
                match bLoFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi - dt_abs * au))
            let add_hi :=
              Tensor.dim (fun i =>
                match bHiFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi + dt_abs * au))
            drs.set! id (some { dim := n, lo := add_lo, hi := add_hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .reshape _ _ | .flatten _ | .concat _ | .transpose _ _ | .permute _
      =>
      match node.parents with
      | #[p1] => drs.set! id (drs[p1]!)
      | _ => drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv .. | .batchNormEval .. => drs
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl propagate init
  else
    init

/--
Propagate first-derivative intervals from a scalar input.

The input derivative is the all-ones vector. The pass uses value-IBP boxes to bound activation
derivatives and leaves an entry empty when it encounters an unsupported local derivative.
-/
def runScalarDerivative
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatBox α)) :=
  runFirstDerivativeWithSeed g ps ibp fun B =>
    if B.dim = 1 then
      let one := Spec.fill (α := α) Numbers.one (.dim B.dim .scalar)
      some { dim := B.dim, lo := one, hi := one }
    else
      none

/--
Propagate a directional first-derivative enclosure from a caller-supplied input seed.

A seed whose dimension differs from an input box leaves that input unresolved. Point seeds such
as coordinate vectors recover partial derivatives; interval seeds propagate a family of
directions through the same local derivative rules.
-/
def runDirectionalDerivative
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (seed : FlatBox α) : Array (Option (FlatBox α)) :=
  runFirstDerivativeWithSeed g ps ibp fun B =>
    if h : seed.dim = B.dim then
      let lo := castDimScalar (α := α) (n := seed.dim) (n' := B.dim) (h := h) seed.lo
      let hi := castDimScalar (α := α) (n := seed.dim) (n' := B.dim) (h := h) seed.hi
      some { dim := B.dim, lo, hi }
    else
      none

/--
Propagate an enclosure of the mixed second derivative `D²f[u, v]`.

`dLeft` and `dRight` are first-derivative passes seeded by directions `u` and `v`. The input mixed
derivative is zero, while every nonlinear rule applies the bilinear second-order chain rule. Taking
the two arrays equal recovers the second directional derivative `D²f[v, v]`; coordinate seeds can
be paired with a fixed direction to recover the entries of a Hessian-vector product.
-/
def runMixedSecondDerivative (g : Graph) (ps : ParamStore α)
    (ibp dLeft dRight : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (d2s : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim B.dim .scalar)
        d2s.set! id (some { dim := B.dim, lo := z, hi := z })
      | none => d2s
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        d2s.set! id (some { dim := v.n, lo := z, hi := z })
      | none => d2s
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      d2s.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool .. | .avgPool .. =>
      -- Not supported by the second-derivative bound pass.
      d2s
    | .hardMaskedSoftmax _ =>
      -- A sound row-wise Hessian rule has not yet been added for masked attention.
      d2s
    | .linear =>
      match node.parents with
      | #[p1] =>
        match d2s[p1]!, ps.linearWB[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .matmul =>
      match node.parents with
      | #[p1] =>
        match d2s[p1]!, ps.matmulW[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (boxAdd (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (boxSub (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .mul_elem =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]!, dLeft[p1]!, dRight[p1]!, dLeft[p2]!, dRight[p2]!,
            d2s[p1]!, d2s[p2]! with
        | some xB, some yB, some dxLeft, some dxRight, some dyLeft, some dyRight,
            some d2x, some d2y =>
          -- D²(xy)[u,v] = D²x[u,v]y + Dx[u]Dy[v] + Dx[v]Dy[u] + xD²y[u,v].
          match boxMulElem (α := α) d2x yB,
              boxMulElem (α := α) dxLeft dyRight,
              boxMulElem (α := α) dxRight dyLeft,
              boxMulElem (α := α) xB d2y with
          | some t1, some t2, some t3, some t4 =>
            d2s.set! id <| some <|
              boxAdd (α := α) (boxAdd (α := α) t1 t2) (boxAdd (α := α) t3 t4)
          | _, _, _, _ => d2s
        | _, _, _, _, _, _, _, _ => d2s
      | _ => d2s
    | .minMax _ => d2s
    | .relu =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]! with
        | some zB =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim zB.dim .scalar)
          d2s.set! id (some { dim := zB.dim, lo := z, hi := z })
        | none => d2s
      | _ => d2s
    | .tanh =>
      match node.parents with
      | #[p1] =>
        match dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some dzLeft, some dzRight, some d2z =>
          match boxMulElem (α := α) dzLeft dzRight with
          | none => d2s
          | some dzProduct =>
            match boxMulElem (α := α)
                (tanhSecondDerivBox (α := α) dzProduct.dim) dzProduct,
              boxMulElem (α := α) (tanhDerivBox (α := α) d2z.dim) d2z with
            | some tA, some tB => d2s.set! id (some (boxAdd (α := α) tA tB))
            | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sin =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some zB, some dzLeft, some dzRight, some d2z =>
          -- D²sin(z)[u,v] = -sin(z) Dz[u] Dz[v] + cos(z) D²z[u,v].
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB,
              boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some sinB, some cosB =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) (boxNeg (α := α) sinB) dzProduct,
                  boxMulElem (α:=α) cosB d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .cos =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some zB, some dzLeft, some dzRight, some d2z =>
          -- D²cos(z)[u,v] = -cos(z) Dz[u] Dz[v] - sin(z) D²z[u,v].
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB,
              boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some sinB, some cosB =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) (boxNeg (α := α) cosB) dzProduct,
                  boxMulElem (α:=α) (boxNeg (α := α) sinB) d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .sigmoid =>
      match node.parents with
      | #[p1] =>
        match dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some dzLeft, some dzRight, some d2z =>
          match boxMulElem (α := α) dzLeft dzRight with
          | none => d2s
          | some dzProduct =>
            match boxMulElem (α := α)
                (sigmoidSecondDerivBox (α := α) dzProduct.dim) dzProduct,
              boxMulElem (α := α) (sigmoidDerivBox (α := α) d2z.dim) d2z with
            | some tA, some tB => d2s.set! id (some (boxAdd (α := α) tA tB))
            | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .exp =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some zB, some dzLeft, some dzRight, some d2z =>
          match derivBoxExp? (α := α) zB with
          | some derivative =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) derivative dzProduct,
                  boxMulElem (α:=α) derivative d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | none => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .log =>
      match node.parents with
      | #[p1] =>
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some zB, some dzLeft, some dzRight, some d2z =>
          match derivBoxLog? (α := α) zB, secondDerivBoxLog? (α := α) zB with
          | some firstDerivative, some secondDerivative =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) secondDerivative dzProduct,
                  boxMulElem (α:=α) firstDerivative d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .sum =>
      match node.parents with
      | #[p1] =>
        match d2s[p1]! with
        | some d2Xin => d2s.set! id (some (boxSum (α := α) d2Xin))
        | none => d2s
      | _ => d2s
    | .reshape _ _ | .flatten _ | .concat _ | .transpose _ _ | .permute _
      =>
      match node.parents with
      | #[p1] => d2s.set! id (d2s[p1]!)
      | _ => d2s
    | .mseLoss => d2s
    | .softmax _ =>
      -- D²y_i[u,v] = Σ_k J_ik D²z_k[u,v] + Σ_{j,k} H_ijk Dz_j[u] Dz_k[v], with
      -- J = diag(y) - y yᵀ and H derived from ∂J/∂z (bounded via y-bounds).
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then d2s else
      match node.parents with
      | #[p1] =>
        match ibp[id]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some yB, some dzLeft, some dzRight, some d2z =>
          -- The Hessian below is for one vector-valued softmax row.
          if hLeft : dzLeft.dim = yB.dim ∧ node.outShape = .dim yB.dim .scalar then
            if hRight : dzRight.dim = yB.dim then
              if h2 : d2z.dim = yB.dim then
              let n := yB.dim
              -- Cast derivative tensors to dimension n for Fin alignment
              let dLeftLo := castDimScalar (α:=α) (n:=dzLeft.dim) (n':=n)
                (h:=hLeft.1) dzLeft.lo
              let dLeftHi := castDimScalar (α:=α) (n:=dzLeft.dim) (n':=n)
                (h:=hLeft.1) dzLeft.hi
              let dRightLo := castDimScalar (α:=α) (n:=dzRight.dim) (n':=n)
                (h:=hRight) dzRight.lo
              let dRightHi := castDimScalar (α:=α) (n:=dzRight.dim) (n':=n)
                (h:=hRight) dzRight.hi
              let d2Lo := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.lo
              let d2Hi := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.hi
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let fdLeftLo := getDimScalarFn (α:=α) dLeftLo
              let fdLeftHi := getDimScalarFn (α:=α) dLeftHi
              let fdRightLo := getDimScalarFn (α:=α) dRightLo
              let fdRightHi := getDimScalarFn (α:=α) dRightHi
              let fd2Lo := getDimScalarFn (α:=α) d2Lo
              let fd2Hi := getDimScalarFn (α:=α) d2Hi
              let mulI (aLo aHi bLo bHi : α) : α × α :=
                let p1 := aLo * bLo; let p2 := aLo * bHi
                let p3 := aHi * bLo; let p4 := aHi * bHi
                let lo1 := if p1 < p2 then p1 else p2
                let lo2 := if p3 < p4 then p3 else p4
                let lo  := if lo1 < lo2 then lo1 else lo2
                let hi1 := if p1 > p2 then p1 else p2
                let hi2 := if p3 > p4 then p3 else p4
                let hi  := if hi1 > hi2 then hi1 else hi2
                (lo, hi)
              -- Bounds for (δ_ik - y_k)
              let deltaMinus (i k : Fin n) : α × α :=
                if decide (i.val = k.val) then
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  (Numbers.one - ykHi, Numbers.one - ykLo)
                else
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  ((-ykHi), (-ykLo))
              -- J*d2z term per i
              let part1_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part1_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              -- Quadratic term Σ_{j,k} H_ijk dz_j dz_k, use interval-bounded H from y-bounds
              let part2_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        -- H_ijk = y_i (dij)(dik) - y_i y_j (δ_jk - y_k)
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        -- H interval = t1 - t2
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fdLeftLo j with | .scalar v => v
                        let dzjHi := match fdLeftHi j with | .scalar v => v
                        let dzkLo := match fdRightLo k with | .scalar v => v
                        let dzkHi := match fdRightHi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part2_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fdLeftLo j with | .scalar v => v
                        let dzjHi := match fdLeftHi j with | .scalar v => v
                        let dzkLo := match fdRightLo k with | .scalar v => v
                        let dzkHi := match fdRightHi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              let lo := Tensor.addSpec part1_lo part2_lo
              let hi := Tensor.addSpec part1_hi part2_hi
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
              else d2s
            else d2s
          else d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .layernorm _ =>
      -- The existing diagonal second-order formula does not establish the mixed bilinear term.
      -- Leave the node unresolved until a row-wise LayerNorm Hessian enclosure is available.
      d2s
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      d2s
    | .conv .. | .batchNormEval .. => d2s
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl propagate init
  else
    init

/-- Propagate the second directional derivative `D²f[v, v]` from one first-derivative pass. -/
def runSecondDirectionalDerivative (g : Graph) (ps : ParamStore α)
    (ibp dDirection : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  runMixedSecondDerivative g ps ibp dDirection dDirection

/--
Compute an interval enclosure for each component of a Hessian-vector product.

`coordinateDerivatives i` is the first-derivative pass seeded by the `i`th coordinate vector;
`directionalDerivative` is seeded by the vector being multiplied by the Hessian. The result at `i`
is the mixed derivative `D²f[eᵢ, v]`, i.e. the `i`th Hessian-vector component for scalar outputs.
-/
def runHessianVectorProduct {inputDim : Nat} (g : Graph) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (coordinateDerivatives : Fin inputDim → Array (Option (FlatBox α)))
    (directionalDerivative : Array (Option (FlatBox α))) :
    Fin inputDim → Array (Option (FlatBox α)) :=
  fun i => runMixedSecondDerivative g ps ibp (coordinateDerivatives i) directionalDerivative

/-- One-dimensional second derivatives are the all-ones directional special case. -/
def runScalarSecondDerivative (g : Graph) (ps : ParamStore α)
    (ibp d1 : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  runSecondDirectionalDerivative g ps ibp d1

end NN.MLTheory.CROWN.Graph
