import Lake
open Lake DSL

package mathlib_atp where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib MathlibAtp where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
