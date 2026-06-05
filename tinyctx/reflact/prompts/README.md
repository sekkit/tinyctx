# ReflACT prompt templates — loaded by reflect.py.
# These are more detailed than the inline fallbacks and include
# structured formatting constraints.
#
# Variables (substituted at runtime):
#   {skill_content}  — current skill document
#   {trajectories}   — formatted rollout trajectories
#   {step_buffer}    — past rejected edits and patterns
#   {meta_skill}     — cross-epoch optimizer guidance
