reuse the skeleton of this code but:

i want C++ with wasm - ts/js binding for an ASE style interface

important goals:

* neighbor list style neighbor search (via vesin?)
* vasp read support
* cif files
* lammps files
* xyz file

i think i would rather reuse some external library like gemmi
here i actually prefer reuse external library
i also think the

modern c++

maybe also python binding with pybind11? - just for test?

this is prepare for intergration of git@github.com:chazeon/gemmy.git (the python part)
so here it provide an interface to do the first stage in browser
actual convex hull logic i currently don't expect it to provide

use uv if necessary

js lib of course - 

make use of multiple subagent - haiku to code, test, run stuff; sonnet to audit

i suspect its handling of vasp edge cases is not there

think about Fe1, Fe_pv1 (things from phonopy)

think about lammps custom atom map / triclinic .... 
think about vasp stuff...

make a plan first then dispatch agents

for the wasm - i want it to be customizable (so cut to what we need)
some burndern like yaml probably not nessary to bundle