## Aim

tnPolicyNet is a transformer encoder with two heads: one chooses the rewrite rule (11 classes), the other the node to apply it at (a softmax over positions). The net decides and the exact engine executes, so every rewrite is sound.

## Where it stands

Drawn in the net-diagram editor, with its weights counted, and its training notebook generated (`tnPolicyNetFromDiagram.nb`). Training data from the TensorNets engine is the next step. The full design is in `DESIGN-tensornets.md`.
