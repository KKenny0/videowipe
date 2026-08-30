# Domain Context

## Clean Planning

The domain process that turns a video and cleanup request into a deterministic
`WipePlan`. It owns request interpretation, detection configuration and
execution, candidate selection, and plan construction/refinement.

Interactive confirmation, progress presentation, artifact persistence, and
inpainting execution stay in their adapters.

## Clean Plan Draft

The reviewable intermediate result of Clean Planning. It contains detected
candidates, the resolved request, and the proposed remove selection together
with the runtime evidence needed for final refinement. It is not executable.

An adapter may present or override the proposed selection. Finalizing the draft
produces the deterministic `WipePlan`.

The draft and its planning interface are internal. `WipeEngine.plan()` remains
the public planning interface.
